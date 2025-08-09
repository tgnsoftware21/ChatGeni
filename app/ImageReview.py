import logging
from flask import Blueprint, Flask, jsonify, request, render_template, send_file
import openai
import base64
import os
import io
from flask import make_response
from xhtml2pdf import pisa
import re
import markdown
import json
from itertools import combinations
from flask import make_response, render_template_string
from datetime import datetime
from werkzeug.utils import secure_filename
from collections import defaultdict
import mimetypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

generatecomparison = Blueprint("generatecomparison", __name__)

# Azure OpenAI Configuration - Use environment variables for security
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

if not API_KEY or not AZURE_ENDPOINT:
    logger.error("Azure OpenAI API key or endpoint not configured!")
    raise ValueError("Missing Azure OpenAI configuration")

client = openai.AzureOpenAI(
    api_key=API_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=AZURE_ENDPOINT,
)

# Configuration constants
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "app/OutputFiles"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def allowed_file(filename):
    """Check if file has allowed extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_file_size(file_stream):
    """Validate file size"""
    file_stream.seek(0, os.SEEK_END)
    size = file_stream.tell()
    file_stream.seek(0)
    return size <= MAX_FILE_SIZE


def encode_image_to_base64(file_path):
    """Convert image file to base64 data URL"""
    try:
        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"

        # Read and encode file
        with open(file_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode("utf-8")

        # Return data URL
        return f"data:{mime_type};base64,{encoded_string}"

    except Exception as e:
        logger.error(f"Failed to encode image {file_path}: {e}")
        return None


def load_base64_images():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where app.py is
    images = {}
    image_files = {
        "logo": os.path.join(BASE_DIR, "static", "img", "logo.png"),
        "approved": os.path.join(BASE_DIR, "static", "img", "Approved.png"),
        "pending": os.path.join(BASE_DIR, "static", "img", "Pending.png"),
        "rejected": os.path.join(BASE_DIR, "static", "img", "Rejected.png"),
    }

    for name, path in image_files.items():
        if os.path.exists(path):
            with open(path, "rb") as img_file:
                base64_data = base64.b64encode(img_file.read()).decode("utf-8")
                ext = os.path.splitext(path)[1][1:]  # file extension without dot
                images[f"{name}_base64"] = f"data:image/{ext};base64,{base64_data}"
        else:
            print(f"Warning: {path} not found.")
            images[f"{name}_base64"] = None  # let template fallback

    return images


# Load images once at startup
BASE64_IMAGES = load_base64_images()


def extract_set_identifier(filename):
    """
    Extract set identifier from filename to group related images.
    This function determines which images belong together for comparison.

    Examples:
    - "kitchen_before.jpg" -> "kitchen"
    - "bathroom_repair_after.png" -> "bathroom_repair"
    - "living_room_1_before.jpg" -> "living_room_1"
    - "IMG_001_kitchen_before.jpg" -> "kitchen"
    - "20240101_bedroom_after.jpg" -> "bedroom"
    """
    # Remove file extension
    name_without_ext = os.path.splitext(filename.lower())[0]

    # Remove common prefixes (timestamps, IMG_, etc.)
    name_clean = re.sub(r"^(img_\d+_|photo_\d+_|\d{8}_|\d{6}_)", "", name_without_ext)

    # Remove BDA type indicators and common suffixes
    type_indicators = [
        "_before",
        "_during",
        "_after",
        "_pre",
        "_post",
        "_initial",
        "_final",
        "_1",
        "_2",
        "_3",
    ]
    for indicator in type_indicators:
        name_clean = name_clean.replace(indicator, "")

    # Remove trailing numbers and underscores
    name_clean = re.sub(r"[_\d]+$", "", name_clean)

    # If nothing meaningful left, use the original filename approach
    if not name_clean or len(name_clean) < 2:
        # Fallback: use first meaningful part of filename
        parts = filename.lower().split("_")
        for part in parts:
            if len(part) > 2 and not part.isdigit():
                return part
        return "default_set"

    return name_clean


def group_images_into_sets(images_data):
    """
    Group images into sets based on filename patterns and return matched pairs.

    Args:
        images_data: List of dicts with 'filename', 'path', 'bda_type_id'

    Returns:
        List of comparison pairs ready for processing
    """
    # Group images by set identifier
    image_sets = defaultdict(lambda: {"before": [], "during": [], "after": []})

    for img_data in images_data:
        set_id = extract_set_identifier(img_data["filename"])
        bda_type_id = img_data["bda_type_id"]

        if bda_type_id == 1:  # Before
            image_sets[set_id]["before"].append(img_data)
        elif bda_type_id == 2:  # During
            image_sets[set_id]["during"].append(img_data)
        elif bda_type_id == 3:  # After
            image_sets[set_id]["after"].append(img_data)

        logger.info(
            f"Added {img_data['filename']} to set '{set_id}' as type {bda_type_id}"
        )

    # Create comparison pairs from sets
    comparison_pairs = []

    for set_id, set_images in image_sets.items():
        logger.info(
            f"Processing set '{set_id}': {len(set_images['before'])} before, {len(set_images['during'])} during, {len(set_images['after'])} after"
        )

        # Strategy 1: Before -> After (most common)
        if set_images["before"] and set_images["after"]:
            for before_img in set_images["before"]:
                for after_img in set_images["after"]:
                    comparison_pairs.append(
                        {
                            "before": before_img,
                            "after": after_img,
                            "set_id": set_id,
                            "comparison_type": "before_after",
                        }
                    )
                    logger.info(
                        f"Created BEFORE->AFTER pair in set '{set_id}': {before_img['filename']} -> {after_img['filename']}"
                    )

        # Strategy 2: Before -> During (if no after images in this set)
        elif set_images["before"] and set_images["during"] and not set_images["after"]:
            for before_img in set_images["before"]:
                for during_img in set_images["during"]:
                    comparison_pairs.append(
                        {
                            "before": before_img,
                            "after": during_img,
                            "set_id": set_id,
                            "comparison_type": "before_during",
                        }
                    )
                    logger.info(
                        f"Created BEFORE->DURING pair in set '{set_id}': {before_img['filename']} -> {during_img['filename']}"
                    )

        # Strategy 3: During -> After (if no before images in this set)
        elif set_images["during"] and set_images["after"] and not set_images["before"]:
            for during_img in set_images["during"]:
                for after_img in set_images["after"]:
                    comparison_pairs.append(
                        {
                            "before": during_img,
                            "after": after_img,
                            "set_id": set_id,
                            "comparison_type": "during_after",
                        }
                    )
                    logger.info(
                        f"Created DURING->AFTER pair in set '{set_id}': {during_img['filename']} -> {after_img['filename']}"
                    )

        # Strategy 4: Three-way comparison (Before -> During -> After)
        elif set_images["before"] and set_images["during"] and set_images["after"]:
            # Create both Before->During and During->After pairs
            for before_img in set_images["before"]:
                for during_img in set_images["during"]:
                    comparison_pairs.append(
                        {
                            "before": before_img,
                            "after": during_img,
                            "set_id": set_id,
                            "comparison_type": "before_during",
                        }
                    )

            for during_img in set_images["during"]:
                for after_img in set_images["after"]:
                    comparison_pairs.append(
                        {
                            "before": during_img,
                            "after": after_img,
                            "set_id": set_id,
                            "comparison_type": "during_after",
                        }
                    )

            logger.info(f"Created 3-way comparison pairs in set '{set_id}'")

        else:
            logger.warning(
                f"Set '{set_id}' has insufficient images for comparison: {dict(set_images)}"
            )

    return comparison_pairs


def encode_image(image_path):
    """Encode image to base64"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        raise


def get_system_prompt():
    """Return the system prompt for image comparison"""
    return """You are an expert AI assistant trained in real estate preservation, REO property management, and field service quality control. You validate work order photos by comparing "Before" and "After" images.

Your goal is to verify:
- Location and context consistency between images
- Work completion quality and compliance
- Image authenticity and manipulation detection
- Repair scope alignment with expected work

⚙ Visual Analysis Process:
- Analyze structural elements, lighting, angles, and backgrounds
- Identify work type and transformation quality
- Detect any signs of image manipulation or duplication
- Evaluate safety and compliance standards

Follow this EXACT output format:

---

🖼 Title: [Brief description of work performed]

🔍 Validation Checklist

| Factor | Analysis | Result |
|--------|----------|--------|
| Location Match | GPS/metadata alignment assessment | ✅ / ❌ |
| Duplicate Check | Images identical or nearly same | ✅ / ❌ |
| Tampering Check | Evidence of editing or manipulation | ✅ / ❌ |
| Area Consistency | Same room/location verification | ✅ / ❌ |
| Work Scope | Expected work type visible | ✅ / ❌ |
| Photo Quality | Clarity, lighting, focus assessment | ✅ / ❌ |

🚰 Feature Comparison

| Feature | Before | After | Score (1-10) |
|---------|--------|-------|-------------|
| Damage Condition | | | |
| Cleanliness | | | |
| Safety Compliance | | | |
| Work Completion | | | |
| Area Identification | | | |
| Visual Consistency | | | |

💵 Estimated Cost: $[amount range]
📈 Confidence: [0-100]%
🧱 Total Score: [0-100]

📟 Verdict:
- ✅ Approved (≥80) / ⚠ Review (50-79) / ❌ Rejected (<50)

📄 Summary:
[3-5 sentences describing changes, quality, and concerns]

---

json
{
  "score": [0-100],
  "verdict": "[status with emoji]",
  "confidence": [0-100],
  "repair_cost": "[range]",
  "features": {
    "damage_condition": {"before": "", "after": "", "score": 0},
    "cleanliness": {"before": "", "after": "", "score": 0},
    "safety_compliance": {"before": "", "after": "", "score": 0},
    "work_completion": {"before": "", "after": "", "score": 0},
    "area_identification": {"before": "", "after": "", "score": 0},
    "visual_consistency": {"before": "", "after": "", "score": 0}
  }
}
"""


def generate_comparison(before_path, after_path):
    """Generate comparison analysis between before and after images"""
    try:
        before_data = encode_image(before_path)
        after_data = encode_image(after_path)

        messages = [
            {"role": "system", "content": get_system_prompt()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Compare these before and after images. Provide detailed analysis with scoring and verdict.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{before_data}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{after_data}"},
                    },
                ],
            },
        ]

        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=2500,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Failed to generate comparison: {e}")
        return f"Error generating comparison: {str(e)}"


@generatecomparison.route("/")
def index():
    """Render upload page"""
    return render_template("upload.html")


@generatecomparison.route("/chatgenie/v1/upload", methods=["POST"])
def upload():
    """Handle image upload and comparison processing with BDATypeId and set-based matching"""
    try:
        uploaded_files = request.files.getlist("images")
        image_types = request.form.getlist("imageTypes")  # Get the BDATypeId values

        if not uploaded_files or all(f.filename == "" for f in uploaded_files):
            return jsonify({"error": "No files uploaded"}), 400

        logger.info(f"Received {len(uploaded_files)} files for processing")
        logger.info(f"Image types: {image_types}")

        # Create directories
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Process and collect all images with their metadata
        all_images_data = []

        # Process each uploaded file
        for idx, file in enumerate(uploaded_files):
            if not file or file.filename == "":
                continue

            if not allowed_file(file.filename):
                return (
                    jsonify({"error": f"File type not allowed: {file.filename}"}),
                    400,
                )

            if not validate_file_size(file):
                return jsonify({"error": f"File too large: {file.filename}"}), 400

            # Save file securely
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{idx:02d}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

            file.save(file_path)
            logger.info(f"Saved file {idx+1}/{len(uploaded_files)}: {file_path}")

            # Get the corresponding BDATypeId (default to 1 if not provided)
            bda_type_id = int(image_types[idx]) if idx < len(image_types) else 1

            # Add to images data
            image_data = {
                "filename": filename,
                "original_filename": filename,
                "path": file_path,
                "bda_type_id": bda_type_id,
                "index": idx,
            }
            all_images_data.append(image_data)

            type_name = {1: "BEFORE", 2: "DURING", 3: "AFTER"}.get(
                bda_type_id, "UNKNOWN"
            )
            logger.info(
                f"Added {filename} as {type_name} image (BDATypeId: {bda_type_id})"
            )

        if len(all_images_data) < 2:
            return jsonify({"error": "Need at least 2 images for comparison"}), 400

        # Group images into sets and create comparison pairs
        logger.info("Grouping images into sets for comparison...")
        comparison_pairs = group_images_into_sets(all_images_data)

        if not comparison_pairs:
            # Log debug information about why no pairs were found
            sets_info = defaultdict(lambda: {"before": 0, "during": 0, "after": 0})
            for img in all_images_data:
                set_id = extract_set_identifier(img["filename"])
                type_name = {1: "before", 2: "during", 3: "after"}.get(
                    img["bda_type_id"], "unknown"
                )
                sets_info[set_id][type_name] += 1

            return (
                jsonify(
                    {
                        "error": "No valid comparison pairs found. Check that images have matching set identifiers and proper before/after types.",
                        "debug": {
                            "total_images": len(all_images_data),
                            "sets_found": dict(sets_info),
                            "example_set_ids": [
                                extract_set_identifier(img["filename"])
                                for img in all_images_data[:5]
                            ],
                        },
                    }
                ),
                400,
            )

        logger.info(f"Found {len(comparison_pairs)} comparison pairs")

        results = []
        all_comparisons_html = []

        # Process each comparison pair
        for pair_index, pair_data in enumerate(comparison_pairs):
            before_img = pair_data["before"]
            after_img = pair_data["after"]
            set_id = pair_data["set_id"]
            comparison_type = pair_data["comparison_type"]

            logger.info(
                f"Processing pair {pair_index + 1}/{len(comparison_pairs)} - Set: '{set_id}' ({comparison_type})"
            )

            try:
                # Generate comparison analysis
                comparison_result = generate_comparison(
                    before_img["path"], after_img["path"]
                )

                # Extract score and structured data
                structured_data = {}
                score = 0
                try:
                    json_match = re.search(r"\{.*?\}", comparison_result, re.DOTALL)
                    if json_match:
                        structured_data = json.loads(json_match.group(0))
                        score = structured_data.get("score", 0)
                    else:
                        logger.warning("No JSON object found in comparison_result")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")

                # Fallback: extract score from text if JSON not found
                if not score:
                    score_match = re.search(
                        r"score\s*[:=]\s*(\d+)", comparison_result, re.IGNORECASE
                    )
                    if score_match:
                        score = int(score_match.group(1))

                # Convert markdown to HTML
                result_html = markdown.markdown(
                    comparison_result, extensions=["fenced_code", "tables", "nl2br"]
                )

                # Determine status based on score
                if score >= 80:
                    status = "approved"
                elif score >= 50:
                    status = "pending"
                else:
                    status = "rejected"

                logger.info(f"  Status: {status} (score: {score})")

                # Convert images to base64 data URLs
                before_base64 = encode_image_to_base64(before_img["path"])
                after_base64 = encode_image_to_base64(after_img["path"])

                if not before_base64 or not after_base64:
                    logger.error(f"Failed to encode images for pair {pair_index + 1}")
                    continue

                # Add result with all metadata
                result_item = {
                    "pair_number": pair_index + 1,
                    "set_id": set_id,
                    "before": before_base64,
                    "after": after_base64,
                    "before_filename": before_img["original_filename"],
                    "after_filename": after_img["original_filename"],
                    "before_type_id": before_img["bda_type_id"],
                    "after_type_id": after_img["bda_type_id"],
                    "comparison_type": comparison_type,
                    "score": score,
                    "status": status,
                    "data": structured_data,
                    "html": result_html,
                }
                results.append(result_item)

                # Store for combined report
                all_comparisons_html.append(
                    {
                        "pair_number": pair_index + 1,
                        "set_id": set_id,
                        "before_path": before_base64,
                        "after_path": after_base64,
                        "html_content": result_html,
                        "score": score,
                        "status": status,
                        "comparison_type": comparison_type,
                        "before_filename": before_img["original_filename"],
                        "after_filename": after_img["original_filename"],
                    }
                )

                logger.info(f"✅ Successfully processed pair {pair_index + 1}")

            except Exception as e:
                logger.error(f"Error processing pair {pair_index + 1}: {e}")
                continue

        if not results:
            return jsonify({"error": "Failed to process any image pairs"}), 500

        # Calculate statistics
        total_score = sum(r["score"] for r in results)
        average_score = total_score / len(results) if results else 0
        approved_count = sum(1 for r in results if r["status"] == "approved")

        # Count images by type
        type_counts = {1: 0, 2: 0, 3: 0}
        for img in all_images_data:
            type_counts[img["bda_type_id"]] += 1

        # Count unique sets
        unique_sets = len(
            set(extract_set_identifier(img["filename"]) for img in all_images_data)
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = (
            f"report_{len(results)}pairs_{unique_sets}sets_{timestamp}.html"
        )
        output_path = os.path.join(OUTPUT_FOLDER, report_filename)

        # Generate report using combined_result.html template
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                render_template(
                    "combined_result.html",
                    comparisons=all_comparisons_html,
                    total_pairs=len(results),
                    average_score=round(average_score, 1),
                    approved_count=approved_count,
                    timestamp=timestamp,
                    upload_count=len(uploaded_files),
                    unique_sets=unique_sets,
                    **BASE64_IMAGES,
                )
            )

        # Response data
        response_data = {
            "success": True,
            "message": f"Successfully processed {len(results)} comparison pair{'s' if len(results) > 1 else ''} from {unique_sets} image set{'s' if unique_sets > 1 else ''}",
            "results": results,
            "total_pairs": len(results),
            "unique_sets": unique_sets,
            "report_type": "combined" if len(results) > 1 else "single",
            "filePath": report_filename,
            "average_score": round(average_score, 1),
            "approved_count": approved_count,
            "image_statistics": {
                "total_uploaded": len(uploaded_files),
                "total_processed": len(all_images_data),
                "before_count": type_counts[1],
                "during_count": type_counts[2],
                "after_count": type_counts[3],
                "unique_sets": unique_sets,
            },
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Upload error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# PDF generation and other routes remain the same...
import re
import io
import os
from bs4 import BeautifulSoup
from flask import send_file, jsonify
from werkzeug.utils import secure_filename
from xhtml2pdf import pisa


@generatecomparison.route("/pdf/<html_name>", methods=["GET"])
def generate_pdf(html_name):
    """Generate PDF from saved HTML report with preserved CSS and images."""
    try:
        safe_filename = secure_filename(html_name)
        if not safe_filename.endswith(".html"):
            safe_filename += ".html"

        file_path = os.path.join(OUTPUT_FOLDER, safe_filename)

        if not os.path.exists(file_path):
            return jsonify({"error": "Report not found"}), 404

        # Read HTML
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # Inline CSS
        for link_tag in soup.find_all("link", {"rel": "stylesheet"}):
            css_path = link_tag.get("href")
            if css_path and not css_path.startswith("http"):
                abs_css_path = os.path.join(OUTPUT_FOLDER, css_path)
                if os.path.exists(abs_css_path):
                    with open(abs_css_path, "r", encoding="utf-8") as css_file:
                        style_tag = soup.new_tag("style")
                        style_tag.string = css_file.read()
                        link_tag.replace_with(style_tag)

        # Fix <img> paths to absolute
        for img_tag in soup.find_all("img"):
            src = img_tag.get("src")
            if src and not src.startswith(("http://", "https://", "data:")):
                abs_img_path = os.path.join(OUTPUT_FOLDER, src)
                if os.path.exists(abs_img_path):
                    img_tag["src"] = abs_img_path

        html_content = str(soup)

        # Convert to PDF
        pdf_buffer = io.BytesIO()
        pisa_result = pisa.CreatePDF(
            src=html_content, dest=pdf_buffer, encoding="utf-8"
        )

        if pisa_result.err:
            logger.error(f"PDF generation failed")
            return jsonify({"error": "PDF generation failed"}), 500

        pdf_buffer.seek(0)
        pdf_filename = safe_filename.replace(".html", ".pdf")

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=False,
            download_name=pdf_filename,
        )

    except Exception as e:
        logger.error(f"PDF conversion error: {e}")
        return jsonify({"error": "Could not generate PDF"}), 500


@generatecomparison.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "image-comparison",
        }
    )


# Error handlers
@generatecomparison.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413


@generatecomparison.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500
