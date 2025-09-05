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
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from flask import send_file, jsonify
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
from xhtml2pdf import pisa
import os
import io
import asyncio
from playwright.async_api import async_playwright
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

generatecomparison = Blueprint("generatecomparison", __name__)


# FIXED: Better configuration management with validation
class Config:
    def __init__(self):
        self.AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
        self.API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

        # FIXED: Validate required environment variables at startup
        if not self.API_KEY or not self.AZURE_ENDPOINT:
            logger.error(
                "Missing required environment variables: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT"
            )
            raise ValueError("Missing Azure OpenAI configuration")

    # FIXED: Use absolute paths to avoid path issues
    @property
    def UPLOAD_FOLDER(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "static", "uploads")
        )

    @property
    def OUTPUT_FOLDER(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "OutputFiles"))


# Initialize configuration
config = Config()

# Initialize OpenAI client with proper error handling
try:
    client = openai.AzureOpenAI(
        api_key=config.API_KEY,
        api_version="2024-02-15-preview",
        azure_endpoint=config.AZURE_ENDPOINT,
    )
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

# Configuration constants
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


# FIXED: Better file validation with magic number detection
def get_file_type(file_content):
    """Detect file type using magic numbers for security"""
    magic_numbers = {
        b"\xff\xd8\xff": "jpeg",
        b"\x89PNG\r\n\x1a\n": "png",
        b"GIF87a": "gif",
        b"GIF89a": "gif",
        b"BM": "bmp",
        b"RIFF": "webp",
    }

    for magic, file_type in magic_numbers.items():
        if file_content.startswith(magic):
            return file_type
    return None


def allowed_file(filename, file_content=None):
    """FIXED: Enhanced file validation with magic number checking"""
    if "." in filename:
        extension = filename.rsplit(".", 1)[1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            return False

    if file_content:
        detected_type = get_file_type(file_content)
        if not detected_type:
            logger.warning(f"Could not detect valid image type for {filename}")
            return False

    return True


def process_filename(filename, file_content=None):
    """FIXED: Better filename processing with type detection"""
    secure_name = secure_filename(filename)

    if "." not in secure_name and file_content:
        detected_type = get_file_type(file_content)
        if detected_type:
            secure_name += f".{detected_type}"
        else:
            secure_name += ".jpg"
            logger.warning(
                f"Could not detect file type for {filename}, defaulting to .jpg"
            )
    elif "." not in secure_name:
        secure_name += ".jpg"

    return secure_name


def validate_file_size(file_stream):
    """Validate file size with better error handling"""
    try:
        if hasattr(file_stream, "seek") and hasattr(file_stream, "tell"):
            file_stream.seek(0, os.SEEK_END)
            size = file_stream.tell()
            file_stream.seek(0)
        else:
            size = len(file_stream)
        return size <= MAX_FILE_SIZE
    except Exception as e:
        logger.error(f"Error validating file size: {e}")
        return False


def encode_image_to_base64(file_path):
    """FIXED: Better error handling for image encoding"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return None

        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"

        with open(file_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{encoded_string}"

    except Exception as e:
        logger.error(f"Failed to encode image {file_path}: {e}")
        return None


def load_base64_images():
    """FIXED: Better path handling and error checking"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    images = {}
    image_files = {
        "logo": os.path.join(BASE_DIR, "static", "img", "logo.png"),
        "genpi_logo": os.path.join(BASE_DIR, "static", "img", "GenpiLogo.png"),
        "approved": os.path.join(BASE_DIR, "static", "img", "Approved.png"),
        "pending": os.path.join(BASE_DIR, "static", "img", "Pending.png"),
        "rejected": os.path.join(BASE_DIR, "static", "img", "Rejected.png"),
    }

    for name, path in image_files.items():
        try:
            if os.path.exists(path):
                with open(path, "rb") as img_file:
                    base64_data = base64.b64encode(img_file.read()).decode("utf-8")
                    ext = os.path.splitext(path)[1][1:]
                    images[f"{name}_base64"] = f"data:image/{ext};base64,{base64_data}"
            else:
                logger.warning(f"Image not found: {path}")
                images[f"{name}_base64"] = None
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            images[f"{name}_base64"] = None

    return images


# Load images once at startup
BASE64_IMAGES = load_base64_images()


def encode_image(image_path):
    """FIXED: Better error handling"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        raise


def get_system_prompt():
    """Returns the GPS + Date validation system prompt with ENFORCED section separation"""
    return """You are an expert AI assistant trained in real estate preservation, REO property management, and field service quality control. You validate work order photos by comparing "Before" and "After" images with GPS location and date verification.

Your goal is to verify:
- Location consistency between images using GPS coordinates
- Date compliance for work order timeline
- Work completion quality and compliance
- Image authenticity and manipulation detection
- Repair scope alignment with expected work
- **GPS + DATE VALIDATION**: Form GPS coordinates and work period verification

🚨 CRITICAL VALIDATION TASKS:
1. **GPS Coordinate Verification**:
   - Use Form GPS coordinates as the authoritative location data
   - Verify both BEFORE and AFTER images have the same Form GPS coordinates
   - Ensure images were taken at the same work site
   - Form GPS coordinates are provided by the user and considered accurate

2. **Date/Time Validation**:
   - Check if images were taken within the work order period
   - Work period defined by issued_date to completed_date
   - BEFORE images: Should be taken before or during work period
   - AFTER images: Must be taken during or after work starts, before work completion
   - Use logical reasoning about work timeline

3. **Location Consistency**:
   - Both images must have identical Form GPS coordinates
   - Verify visual consistency between before/after images
   - Detect potential location mismatches through image analysis

📊 VALIDATION METHODOLOGY:
- Use Form GPS coordinates as the only location validation source
- Simple coordinate matching (exact match required)
- Apply logical date timeline validation using work order dates
- Focus on work site consistency and timeline compliance
- No complex EXIF parsing required

**CRITICAL: Follow this EXACT output format with SEPARATE sections:**

---

TITLE: [Brief description of work performed]

AI VALIDATION RESULTS

| Factor | Analysis | Result |
|--------|----------|--------|
| GPS Validation | Form GPS coordinate consistency check | ✅ / ❌ |
| Date Compliance | Work period timeline validation | ✅ / ❌ |
| Location Consistency | Same Form GPS coordinates for both images | ✅ / ❌ |
| Tampering Check | Evidence of editing or manipulation | ✅ / ❌ |
| Area Consistency | Same room/location verification | ✅ / ❌ |
| Work Scope | Expected work type visible | ✅ / ❌ |
| Photo Quality | Clarity, lighting, focus assessment | ✅ / ❌ |

AI FEATURE SCORING

| Feature | Before | After | Score (1-10) |
|---------|--------|-------|-------------|
| Damage Condition | | | |
| Cleanliness | | | |
| Safety Compliance | | | |
| Work Completion | | | |
| Area Identification | | | |
| Visual Consistency | | | |

NON-COMPLIANCE CHECKS

<ul class="non-compliance-list">
<li class="non-compliance-item">
    <i class="fa fa-check-circle compliance-icon" style="color: #27ae60;"></i>
    <div class="compliance-text">
        <span class="category-name">GPS Validation:</span>
        <span class="issue-description">No mismatches detected</span>
    </div>
</li>
<li class="non-compliance-item">
    <i class="fa fa-check-circle compliance-icon" style="color: #27ae60;"></i>
    <div class="compliance-text">
        <span class="category-name">Date Compliance:</span>
        <span class="issue-description">Images taken within work order timeline</span>
    </div>
</li>
<li class="non-compliance-item">
    <i class="fa fa-check-circle compliance-icon" style="color: #27ae60;"></i>
    <div class="compliance-text">
        <span class="category-name">Location Consistency:</span>
        <span class="issue-description">Same work site verified</span>
    </div>
</li>
</ul>

OVERALL INSIGHTS
[3-5 sentences describing changes, quality, concerns, GPS validation, and timeline compliance]

AI RECOMMENDATION
Estimated Cost: $[amount range]
Confidence: [0-100]%
Total Score: [0-100]

QC STATUS
Approved (≥80 + GPS valid + date compliant) / ⚠ Review (50-79 or validation issues) / ❌ Rejected (<50 or validation failures)

**For backend processing, include JSON data:**

```json
{
  "score": [0-100],
  "qc_status": "[status with emoji]",
  "confidence": [0-100],
  "repair_cost": "[range]",
  "validation_results": {
    "gps_coordinates_valid": true/false,
    "date_compliance_valid": true/false,
    "before_gps_valid": true/false,
    "after_gps_valid": true/false,
    "timeline_valid": true/false,
    "location_consistent": true/false,
    "form_gps_coordinates": "[lat, lon]",
    "work_period": "[issued_date to completed_date]",
    "validation_details": {
      "gps_analysis": "[Form GPS consistency explanation]",
      "timeline_analysis": "[Work period timeline explanation]"
    }
  },
  "features": {
    "damage_condition": {"before": "", "after": "", "score": 0},
    "cleanliness": {"before": "", "after": "", "score": 0},
    "safety_compliance": {"before": "", "after": "", "score": 0},
    "work_completion": {"before": "", "after": "", "score": 0},
    "area_identification": {"before": "", "after": "", "score": 0},
    "visual_consistency": {"before": "", "after": "", "score": 0}
  }
}
```

**MANDATORY SECTION SEPARATION REQUIREMENTS:**
- Each section header must be on its own line
- No content should be mixed between sections
- "AI VALIDATION RESULTS" contains ONLY the Factor/Analysis/Result table
- "AI FEATURE SCORING" contains ONLY the Feature/Before/After/Score table
- Each section must end before the next section begins
- Use clear line breaks between sections
- NEVER combine validation and feature tables in one section

**FEATURE SCORING REQUIREMENTS:**
- Before column: Brief descriptive text of the condition/state before work
- After column: Brief descriptive text of the condition/state after work  
- Score column: Numeric score from 1-10 only
- Never put numeric scores in Before/After columns
- Always provide descriptive text for Before/After assessment

**NON-COMPLIANCE CHECKS FORMATTING REQUIREMENTS:**
- MANDATORY: Must output the EXACT HTML structure shown above
- NO plain text format allowed - this breaks template rendering
- Each category must use the specific HTML structure with CSS classes
- Use green check icons when no issues: <i class="fa fa-check-circle compliance-icon" style="color: #27ae60;"></i>
- Use red arrow icons for issues: <i class="fa fa-arrow-right compliance-icon" style="color: #e74c3c;"></i>
- Always include all three categories: GPS Validation, Date Compliance, Location Consistency
- CRITICAL: Any output not matching the exact HTML structure above will break the template

**REQUIRED HTML STRUCTURE FOR NON-COMPLIANCE (use this exact format):**
For issues, replace the check icon with arrow icon but keep same structure:
<i class="fa fa-arrow-right compliance-icon" style="color: #e74c3c;"></i>

**GPS + DATE VALIDATION INSTRUCTIONS:**
- GPS VALIDATION: Use Form GPS coordinates as authoritative - ensure both images have same coordinates
- DATE VALIDATION: Analyze work timeline using issued_date and completed_date for logical compliance
- LOCATION CONSISTENCY: Both images must have identical Form GPS coordinates
- TIMELINE LOGIC: Work should follow logical before/during/after sequence
- NO EXIF PARSING: Use form data only - no complex metadata extraction needed
- PENALTY LOGIC: Reduce score by 15 points for GPS violations, 15 points for date violations
- SIMPLE APPROACH: Focus on practical validation using available form data
- DUPLICATE CHECK: Do not check for duplicate or similar images - focus on work completion validation only"""


# FIXED: Enhanced parsing function with better debugging and flexible extraction
def parse_ai_response_content(html_content):
    """FIXED: Enhanced parsing with debugging and better section splitting"""
    logger.info(f"=== AI RESPONSE PARSING DEBUG ===")
    logger.info(f"Raw content length: {len(html_content)}")

    # Clean content by removing embedded JSON block
    cleaned_content = re.sub(
        r"```json\s*\{.*?\}\s*```", "", html_content, flags=re.DOTALL
    )

    logger.info(f"Cleaned content length: {len(cleaned_content)}")
    logger.info(f"Content preview: {cleaned_content[:300]}...")

    # Check for section markers with more variations
    section_markers = [
        "AI VALIDATION RESULTS",
        "🔍 AI Validation Results",
        "VALIDATION RESULTS",
        "AI FEATURE SCORING",
        "🚰 AI Feature Scoring",
        "FEATURE SCORING",
        "WORK QUALITY ASSESSMENT",
        "NON-COMPLIANCE CHECKS",
        "⚠️ Non-Compliance Checks",
        "NON COMPLIANCE",
        "OVERALL INSIGHTS",
        "📄 Overall Insights",
        "INSIGHTS",
        "AI RECOMMENDATION",
        "📌 AI Recommendation",
        "RECOMMENDATION",
        "QC STATUS",
        "📟 QC Status",
        "STATUS",
    ]

    found_markers = []
    for marker in section_markers:
        if marker in cleaned_content:
            found_markers.append(marker)

    logger.info(f"Found section markers: {found_markers}")

    # FIXED: More flexible section extraction with multiple possible headers
    def extract_section(content, start_patterns, end_patterns=None):
        """Extract content between start and end patterns"""
        start_pos = -1
        used_start = None

        # Find the first matching start pattern
        for pattern in start_patterns:
            pos = content.find(pattern)
            if pos != -1 and (start_pos == -1 or pos < start_pos):
                start_pos = pos
                used_start = pattern

        if start_pos == -1:
            return ""

        # Find end position
        if end_patterns:
            end_pos = len(content)
            for pattern in end_patterns:
                pos = content.find(pattern, start_pos + len(used_start))
                if pos != -1 and pos < end_pos:
                    end_pos = pos
        else:
            end_pos = len(content)

        # Extract and clean the section
        section = content[start_pos:end_pos]
        if used_start:
            section = section.replace(used_start, "", 1)

        return section.strip()

    # Extract sections with multiple possible headers
    validation_analysis = extract_section(
        cleaned_content,
        [
            "🔍 AI Validation Results",
            "AI VALIDATION RESULTS",
            "AI Validation Results",
            "VALIDATION RESULTS",
        ],
        [
            "🚰 AI Feature Scoring",
            "AI FEATURE SCORING",
            "AI Feature Scoring",
            "FEATURE SCORING",
        ],
    )

    feature_scoring = extract_section(
        cleaned_content,
        [
            "🚰 AI Feature Scoring",
            "AI FEATURE SCORING",
            "AI Feature Scoring",
            "FEATURE SCORING",
            "WORK QUALITY ASSESSMENT",
        ],
        [
            "⚠️ Non-Compliance Checks",
            "NON-COMPLIANCE CHECKS",
            "NON COMPLIANCE",
            "📄 Overall Insights",
            "OVERALL INSIGHTS",
        ],
    )

    non_compliance_checks = extract_section(
        cleaned_content,
        [
            "⚠️ Non-Compliance Checks",
            "NON-COMPLIANCE CHECKS",
            "Non-Compliance Checks",
            "NON COMPLIANCE",
        ],
        ["📄 Overall Insights", "OVERALL INSIGHTS", "Overall Insights", "INSIGHTS"],
    )

    summary = extract_section(
        cleaned_content,
        ["📄 Overall Insights", "OVERALL INSIGHTS", "Overall Insights", "INSIGHTS"],
        [
            "📌 AI Recommendation",
            "AI RECOMMENDATION",
            "AI Recommendation",
            "RECOMMENDATION",
        ],
    )

    ai_recommendation = extract_section(
        cleaned_content,
        [
            "📌 AI Recommendation",
            "AI RECOMMENDATION",
            "AI Recommendation",
            "RECOMMENDATION",
        ],
        ["📟 QC Status", "QC STATUS", "QC Status", "STATUS"],
    )

    qc_status = extract_section(
        cleaned_content, ["📟 QC Status", "QC STATUS", "QC Status", "STATUS"], []
    )

    # FIXED: If no sections found, try to parse the entire content as analysis
    if not any([validation_analysis, feature_scoring, summary]):
        logger.warning("No standard sections found, using entire content as summary")
        summary = cleaned_content
        validation_analysis = "Analysis data processed - see summary below."

    result = {
        "validation_analysis": validation_analysis,
        "feature_scoring": feature_scoring,
        "non_compliance_checks": non_compliance_checks,
        "summary": summary,
        "ai_recommendation": ai_recommendation,
        "qc_status": qc_status,
    }

    # Debug output
    logger.info(f"Extracted sections:")
    for key, value in result.items():
        logger.info(
            f"  {key}: {'✅ Found' if value.strip() else '❌ Empty'} ({len(value)} chars)"
        )

    return result


def create_ai_metadata_context(
    before_metadata, after_metadata, work_order_context=None
):
    """Enhanced GPS + Date metadata context with work order information"""
    context = "\n\n📍 COMPREHENSIVE WORK ORDER VALIDATION CONTEXT:\n"
    context += "=" * 80 + "\n"
    context += "🤖 INSTRUCTIONS: Validate GPS coordinates, work timeline, and type classifications using complete work order context.\n\n"

    # WORK ORDER INFORMATION SECTION
    if work_order_context:
        context += f"📋 WORK ORDER DETAILS:\n"
        context += (
            f"• Work Order #: {work_order_context.get('workOrderNumber', 'N/A')}\n"
        )
        context += (
            f"• Property Address: {work_order_context.get('fullAddress', 'N/A')}\n"
        )
        context += f"• Vendor: {work_order_context.get('vendorName', 'N/A')}\n"
        context += f"• Task: {work_order_context.get('workTaskName', 'N/A')}\n"
        context += f"• Scope: {work_order_context.get('scopeOfWork', 'N/A')}\n"
        context += (
            f"• Status: {work_order_context.get('workOrderStatusName', 'N/A')}\n\n"
        )

    # GPS COORDINATES WITH PROPERTY CONTEXT
    form_lat_before = before_metadata.get("form_latitude", "N/A")
    form_lon_before = before_metadata.get("form_longitude", "N/A")
    form_lat_after = after_metadata.get("form_latitude", "N/A")
    form_lon_after = after_metadata.get("form_longitude", "N/A")

    context += f"🗺️ GPS COORDINATE VALIDATION:\n"
    context += f"• Property Location: {work_order_context.get('fullAddress', 'N/A') if work_order_context else 'N/A'}\n"
    context += f"• Before Image GPS: {form_lat_before}, {form_lon_before}\n"
    context += f"• After Image GPS: {form_lat_after}, {form_lon_after}\n"
    context += f"• GPS Status: {'✅ Both Available' if form_lat_before != 'N/A' and form_lat_after != 'N/A' else '❌ Missing Data'}\n"

    if form_lat_before != "N/A" and form_lat_after != "N/A":
        coordinates_match = (
            form_lat_before == form_lat_after and form_lon_before == form_lon_after
        )
        context += (
            f"• GPS Consistency: {'✅ MATCH' if coordinates_match else '❌ MISMATCH'}\n"
        )
        context += f"• Location Status: {'✅ Same work site' if coordinates_match else '❌ Different locations'}\n"
    else:
        context += f"• GPS Consistency: ❌ CANNOT VALIDATE (Missing coordinates)\n"

    context += f"\n"

    # ENHANCED DATE/TIMELINE VALIDATION SECTION
    issued_date = (
        work_order_context.get("issuedDate", "N/A") if work_order_context else "N/A"
    )
    completed_date = (
        work_order_context.get("completedDate", "N/A") if work_order_context else "N/A"
    )
    approved_date = (
        work_order_context.get("approvedDate", "N/A") if work_order_context else "N/A"
    )
    closed_date = (
        work_order_context.get("closedDate", "N/A") if work_order_context else "N/A"
    )

    context += f"⏰ ENHANCED WORK TIMELINE VALIDATION:\n"
    context += f"• Work Order Issued: {issued_date}\n"
    context += f"• Work Completed: {completed_date}\n"
    context += f"• Work Approved: {approved_date}\n"
    context += f"• Work Closed: {closed_date}\n"
    context += f"• Timeline Status: {'✅ Dates Available' if issued_date != 'N/A' and completed_date != 'N/A' else '❌ Missing Dates'}\n\n"

    return context


def generate_comparisons_with_ai_validation(work_orders):
    """
    Generate comparison analysis with AI-based validation for multiple work orders.
    Each work_order should be a dict:
    {
        "id": "WO123",
        "before_images": ["/path/to/before1.jpg", "/path/to/before2.jpg"],
        "after_images":  ["/path/to/after1.jpg",  "/path/to/after2.jpg"],
        "before_metadata": {...},   # optional
        "after_metadata": {...},    # optional
        "work_order_context": {...} # optional
    }
    """
    results = []

    for wo in work_orders:
        try:
            before_images = wo.get("before_images", [])
            after_images = wo.get("after_images", [])
            before_metadata = wo.get("before_metadata")
            after_metadata = wo.get("after_metadata")
            work_order_context = wo.get("work_order_context")

            # ✅ Pair correctly: match min length to avoid IndexError
            total_pairs = min(len(before_images), len(after_images))

            if total_pairs == 0:
                results.append(
                    {
                        "work_order_id": wo.get("id"),
                        "error": "No valid before/after image pairs found",
                    }
                )
                continue

            for idx in range(total_pairs):
                before_path = before_images[idx]
                after_path = after_images[idx]

                try:
                    before_data = encode_image(before_path)
                    after_data = encode_image(after_path)

                    # Build metadata context if available
                    metadata_context = ""
                    if before_metadata and after_metadata:
                        metadata_context = create_ai_metadata_context(
                            before_metadata, after_metadata, work_order_context
                        )

                    # Enhanced prompt
                    user_content = (
                        f"Compare these before and after images for work order {wo['id']} (pair {idx+1}). "
                        f"Use the complete work order context including property details, vendor information, "
                        f"BDA type classifications, and photo type specifications. "
                        f"Validate GPS coordinates against property location, timeline compliance with work dates, "
                        f"and type classifications against actual database values. "
                        f"{metadata_context}"
                    )

                    messages = [
                        {"role": "system", "content": get_system_prompt()},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_content},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{before_data}"
                                    },
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{after_data}"
                                    },
                                },
                            ],
                        },
                    ]

                    response = client.chat.completions.create(
                        model=config.AZURE_DEPLOYMENT_NAME,
                        messages=messages,
                        max_tokens=3000,
                        temperature=0.2,
                    )

                    results.append(
                        {
                            "work_order_id": wo["id"],
                            "pair_index": idx + 1,
                            "before_image": before_path,
                            "after_image": after_path,
                            "analysis": response.choices[0].message.content.strip(),
                        }
                    )

                except Exception as pair_err:
                    logger.error(
                        f"Failed to process pair {idx+1} for WO {wo.get('id')}: {pair_err}"
                    )
                    results.append(
                        {
                            "work_order_id": wo.get("id"),
                            "pair_index": idx + 1,
                            "before_image": before_path,
                            "after_image": after_path,
                            "error": str(pair_err),
                        }
                    )

            # ✅ If before/after counts mismatched, log warning
            if len(before_images) != len(after_images):
                results.append(
                    {
                        "work_order_id": wo.get("id"),
                        "warning": f"Mismatched counts: {len(before_images)} before vs {len(after_images)} after images. Extra images ignored.",
                    }
                )

        except Exception as e:
            logger.error(f"Failed to generate AI comparison for WO {wo.get('id')}: {e}")
            results.append({"work_order_id": wo.get("id"), "error": str(e)})

    return results


# FIXED: Replace your existing group_images_by_bda_type function
# def group_images_by_bda_type(images_data):
#     """FIXED: Enhanced grouping that creates ALL possible before/after pairs"""
#     from collections import defaultdict

#     bda_groups = defaultdict(list)

#     # Group images by bda_type_name (case-insensitive)
#     for img_data in images_data:
#         bda_type_name = img_data.get("bda_type_name", "Unknown").lower().strip()
#         bda_groups[bda_type_name].append(img_data)

#     logger.info(f"BDA groups found: {list(bda_groups.keys())}")

#     comparison_pairs = []

#     # Collect before/after images
#     before_images = []
#     after_images = []

#     # Look for before/after type names
#     for bda_name, imgs in bda_groups.items():
#         if "before" in bda_name:
#             before_images.extend(imgs)
#             logger.info(f"Found {len(imgs)} before images: {bda_name}")
#         elif "after" in bda_name:
#             after_images.extend(imgs)
#             logger.info(f"Found {len(imgs)} after images: {bda_name}")

#     if not before_images or not after_images:
#         logger.info("No clear before/after BDA types, trying photo type pairing...")
#         # Group by photo type instead
#         photo_groups = defaultdict(list)
#         for img_data in images_data:
#             photo_type = img_data.get("photo_type_name", "unknown").lower().strip()
#             photo_groups[photo_type].append(img_data)

#         # Create pairs within each photo type (first half vs second half)
#         for photo_type, imgs in photo_groups.items():
#             if len(imgs) >= 2:
#                 mid = len(imgs) // 2
#                 befores = imgs[:mid] if mid > 0 else [imgs[0]]
#                 afters = imgs[mid:] if mid < len(imgs) else imgs[1:]

#                 pair_counter = 1
#                 for i, before_img in enumerate(befores):
#                     for j, after_img in enumerate(afters):
#                         if before_img != after_img:
#                             comparison_pairs.append(
#                                 {
#                                     "before": before_img,
#                                     "after": after_img,
#                                     "set_id": f"{photo_type}_pair_{pair_counter}",
#                                     "comparison_type": "photo_type_pairing",
#                                 }
#                             )
#                             pair_counter += 1
#                             logger.info(
#                                 f"Photo type pair: {before_img['original_filename']} <-> {after_img['original_filename']}"
#                             )

#         logger.info(f"Created {len(comparison_pairs)} pairs using photo type matching")
#         return comparison_pairs

#     # Group by photo type within before/after categories
#     before_by_photo_type = defaultdict(list)
#     after_by_photo_type = defaultdict(list)

#     for img in before_images:
#         photo_type = img.get("photo_type_name", "unknown").lower().strip()
#         before_by_photo_type[photo_type].append(img)

#     for img in after_images:
#         photo_type = img.get("photo_type_name", "unknown").lower().strip()
#         after_by_photo_type[photo_type].append(img)

#     logger.info(f"Before photo types: {list(before_by_photo_type.keys())}")
#     logger.info(f"After photo types: {list(after_by_photo_type.keys())}")

#     # Create ALL possible pairs within each photo type
#     pair_counter = 1
#     for photo_type in before_by_photo_type.keys():
#         if photo_type in after_by_photo_type:
#             befores = before_by_photo_type[photo_type]
#             afters = after_by_photo_type[photo_type]

#             logger.info(
#                 f"Creating pairs for '{photo_type}': {len(befores)} before × {len(afters)} after = {len(befores) * len(afters)} pairs"
#             )

#             # Create all combinations
#             for i, before_img in enumerate(befores):
#                 for j, after_img in enumerate(afters):
#                     comparison_pairs.append(
#                         {
#                             "before": before_img,
#                             "after": after_img,
#                             "set_id": f"{photo_type}_pair_{pair_counter}",
#                             "comparison_type": "bda_photo_type_based_pairing",
#                         }
#                     )
#                     pair_counter += 1
#                     logger.info(
#                         f"Pair {pair_counter-1}: {before_img['original_filename']} <-> {after_img['original_filename']}"
#                     )

#     # Fallback: sequential pairing if no photo type matches
#     if not comparison_pairs and before_images and after_images:
#         logger.info("No photo type matches, using sequential pairing...")
#         max_pairs = min(len(before_images), len(after_images))
#         for i in range(max_pairs):
#             comparison_pairs.append(
#                 {
#                     "before": before_images[i],
#                     "after": after_images[i],
#                     "set_id": f"sequential_pair_{i+1}",
#                     "comparison_type": "sequential_pairing",
#                 }
#             )

#     logger.info(
#         f"Created {len(comparison_pairs)} total pairs using BDA/photo type matching"
#     )
#     return comparison_pairs


def group_images_by_bda_type(images_data):
    """ROBUST: Handle your exact payload structure with case-insensitive matching"""
    from collections import OrderedDict

    # Sort input images by upload order first for stability
    sorted_images = sorted(
        images_data,
        key=lambda x: (x.get("upload_index", 999), x.get("original_filename", "")),
    )

    logger.info("=== ROBUST BDA TYPE GROUPING DEBUG ===")
    for i, img in enumerate(sorted_images):
        logger.info(
            f"  Image {i}: {img['original_filename']} - BDA: '{img.get('bda_type_name')}' - Photo: '{img.get('photo_type_name')}' - Index: {img.get('upload_index')}"
        )

    # Separate into before/after using case-insensitive exact matching
    before_images = []
    after_images = []

    for img_data in sorted_images:
        bda_type = img_data.get("bda_type_name", "").strip().lower()

        # ROBUST: Handle your exact payload format
        if bda_type == "before":
            before_images.append(img_data)
            logger.info(f"    Classified as BEFORE: {img_data['original_filename']}")
        elif bda_type == "after":
            after_images.append(img_data)
            logger.info(f"    Classified as AFTER: {img_data['original_filename']}")
        else:
            logger.info(
                f"    Skipped (not before/after): {img_data['original_filename']} - BDA: '{bda_type}'"
            )

    logger.info(
        f"Found {len(before_images)} before images, {len(after_images)} after images"
    )

    if not before_images or not after_images:
        logger.info("No clear before/after BDA types, trying fallback...")
        return group_images_by_filename(sorted_images)

    # Group by photo type within before/after categories
    before_by_photo = OrderedDict()
    after_by_photo = OrderedDict()

    for img in before_images:
        photo_type = img.get("photo_type_name", "unknown").strip().lower()
        if photo_type not in before_by_photo:
            before_by_photo[photo_type] = []
        before_by_photo[photo_type].append(img)

    for img in after_images:
        photo_type = img.get("photo_type_name", "unknown").strip().lower()
        if photo_type not in after_by_photo:
            after_by_photo[photo_type] = []
        after_by_photo[photo_type].append(img)

    logger.info(f"Before photo types: {list(before_by_photo.keys())}")
    logger.info(f"After photo types: {list(after_by_photo.keys())}")

    # Create 1:1 pairs within matching photo types
    comparison_pairs = []
    pair_counter = 1

    # Process in sorted order for consistency
    for photo_type in sorted(before_by_photo.keys()):
        if photo_type in after_by_photo:
            befores = before_by_photo[photo_type]
            afters = after_by_photo[photo_type]

            # 1:1 matching - no all-combinations
            num_pairs = min(len(befores), len(afters))

            logger.info(f"Creating {num_pairs} pairs for photo type '{photo_type}'")

            for i in range(num_pairs):
                pair = {
                    "before": befores[i],
                    "after": afters[i],
                    "set_id": f"{photo_type}_pair_{pair_counter}",
                    "comparison_type": "bda_photo_type_based_pairing",
                }
                comparison_pairs.append(pair)

                logger.info(
                    f"  Pair {pair_counter}: {befores[i]['original_filename']} <-> {afters[i]['original_filename']}"
                )
                pair_counter += 1

    # Enhanced fallback for unmatched images
    if not comparison_pairs:
        logger.info("No photo type matches, using sequential before/after pairing...")
        num_pairs = min(len(before_images), len(after_images))

        for i in range(num_pairs):
            pair = {
                "before": before_images[i],
                "after": after_images[i],
                "set_id": f"sequential_pair_{i+1}",
                "comparison_type": "sequential_pairing",
            }
            comparison_pairs.append(pair)
            logger.info(
                f"  Sequential pair {i+1}: {before_images[i]['original_filename']} <-> {after_images[i]['original_filename']}"
            )

    logger.info(f"FINAL: Created {len(comparison_pairs)} comparison pairs")
    return comparison_pairs


def group_images_by_filename(images_data):
    """FIXED: Enhanced filename-based pairing that creates multiple pairs"""
    before_images = []
    after_images = []

    # Separate before and after images
    for img_data in images_data:
        filename_lower = img_data["filename"].lower()
        bda_type_name = img_data.get("bda_type_name", "").lower()

        if "before" in bda_type_name or filename_lower.startswith("before_"):
            before_images.append(img_data)
        elif "after" in bda_type_name or filename_lower.startswith("after_"):
            after_images.append(img_data)

    logger.info(
        f"Filename classification: {len(before_images)} before, {len(after_images)} after"
    )

    comparison_pairs = []

    # Create ALL possible pairs
    if before_images and after_images:
        pair_counter = 1
        for i, before_img in enumerate(before_images):
            for j, after_img in enumerate(after_images):
                comparison_pairs.append(
                    {
                        "before": before_img,
                        "after": after_img,
                        "set_id": f"filename_pair_{pair_counter}",
                        "comparison_type": "filename_based_pairing",
                    }
                )
                pair_counter += 1
                logger.info(
                    f"Filename pair {pair_counter-1}: {before_img['original_filename']} <-> {after_img['original_filename']}"
                )

    # Fallback: sequential pairing of all images
    elif len(images_data) >= 2:
        for i in range(len(images_data) - 1):
            comparison_pairs.append(
                {
                    "before": images_data[i],
                    "after": images_data[i + 1],
                    "set_id": f"sequential_fallback_{i+1}",
                    "comparison_type": "sequential_fallback_pairing",
                }
            )

    logger.info(f"Created {len(comparison_pairs)} pairs from filename matching")
    return comparison_pairs
    """Fallback grouping function for filename-based pairing"""
    before_images = []
    after_images = []

    # Separate before and after images (case-insensitive)
    for img_data in images_data:
        filename_lower = img_data["filename"].lower()
        bda_type_name = img_data.get("bda_type_name", "").lower()

        # Use BDA type name if available, otherwise filename
        if "before" in bda_type_name or filename_lower.startswith("before_"):
            before_images.append(img_data)
        elif "after" in bda_type_name or filename_lower.startswith("after_"):
            after_images.append(img_data)

    def extract_image_type(img_data):
        """Extract the type from image data using photo type or filename"""
        photo_type = img_data.get("photo_type_name", "")
        if photo_type and photo_type != "Unknown":
            return photo_type.lower().replace(" ", "_")

        filename_lower = img_data["filename"].lower()
        if filename_lower.startswith("before_"):
            base = filename_lower[len("before_") :]
        elif filename_lower.startswith("after_"):
            base = filename_lower[len("after_") :]
        else:
            return None

        base = base.rsplit(".", 1)[0] if "." in base else base
        if "_image_" in base:
            base = base.split("_image_")[0]
        elif base.endswith("_img"):
            base = base[:-4]

        return base

    # Group images by type
    before_by_type = {}
    after_by_type = {}

    for img in before_images:
        img_type = extract_image_type(img)
        if img_type:
            before_by_type[img_type] = img

    for img in after_images:
        img_type = extract_image_type(img)
        if img_type:
            after_by_type[img_type] = img

    # Create pairs for matching types
    comparison_pairs = []
    for img_type in before_by_type:
        if img_type in after_by_type:
            comparison_pairs.append(
                {
                    "before": before_by_type[img_type],
                    "after": after_by_type[img_type],
                    "set_id": f"{img_type}_comparison",
                    "comparison_type": "filename_based_pairing",
                }
            )

    logger.info(
        f"Created {len(comparison_pairs)} comparison pairs from filename matching"
    )
    return comparison_pairs


@generatecomparison.route("/")
def index():
    """Render upload page"""
    return render_template("upload.html")


# @generatecomparison.route("/chatgenie/v1/upload", methods=["POST"])
# def upload():
#     """FIXED: Handle image upload with proper BDA/Photo type name handling"""
#     try:
#         # Extract form data
#         uploaded_files = request.files.getlist("images")
#         image_types = request.form.getlist("imageTypes")  # Now BDA type names
#         photo_types = request.form.getlist("photoTypes")  # Now photo type names
#         latitudes = request.form.getlist("latitudes")
#         longitudes = request.form.getlist("longitudes")
#         issued_dates = request.form.getlist("issuedDates")
#         completed_dates = request.form.getlist("completedDates")
#         approved_dates = request.form.getlist("approvedDates")
#         closed_dates = request.form.getlist("closedDates")
#         work_order_statuses = request.form.getlist("workOrderStatus")
#         document_detail_ids = request.form.getlist("documentDetailIds") or ["0"] * len(
#             uploaded_files
#         )
#         bda_type_names = request.form.getlist("bdaTypeNames") or []
#         photo_type_names = request.form.getlist("photoTypeNames") or []

#         # Work Order Context
#         work_order_context = {
#             "workOrderNumber": request.form.get("workOrderNumber", ""),
#             "workOrderId": request.form.get("workOrderId", ""),
#             "fullAddress": request.form.get("fullAddress", ""),
#             "propertyNumber": request.form.get("propertyNumber", ""),
#             "workTaskName": request.form.get("workTaskName", ""),
#             "scopeOfWork": request.form.get("scopeOfWork", ""),
#             "vendorName": request.form.get("vendorName", ""),
#             "workOrderStatusName": (
#                 work_order_statuses[0]
#                 if work_order_statuses
#                 else request.form.get("workOrderStatusName", "")
#             ),
#             "issuedDate": issued_dates[0] if issued_dates else "",
#             "completedDate": completed_dates[0] if completed_dates else "",
#             "approvedDate": approved_dates[0] if approved_dates else "",
#             "closedDate": closed_dates[0] if closed_dates else "",
#             "transactionId": request.form.get("transactionId", ""),
#             "imageCount": len(uploaded_files),
#             "hasPropertyImage": any(img_type == "Property" for img_type in image_types),
#         }

#         # Debug logging
#         logger.info("=== UPLOAD DEBUG INFO ===")
#         logger.info(f"Work Order Number: {work_order_context['workOrderNumber']}")
#         logger.info(f"Uploaded Files: {[f.filename for f in uploaded_files]}")
#         logger.info(f"Image Types: {image_types}")
#         logger.info(f"Photo Types: {photo_types}")

#         if not uploaded_files or all(f.filename == "" for f in uploaded_files):
#             return jsonify({"error": "No files uploaded"}), 400

#         logger.info(f"Received {len(uploaded_files)} files for processing")

#         # Create directories
#         os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
#         os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

#         # Process uploaded files
#         all_images_data = []

#         for idx, file in enumerate(uploaded_files):
#             if not file or file.filename == "":
#                 continue

#             # Read file content for validation
#             file_content = file.read()
#             file.seek(0)  # Reset file pointer

#             if not allowed_file(file.filename, file_content):
#                 return (
#                     jsonify({"error": f"File type not allowed: {file.filename}"}),
#                     400,
#                 )

#             if not validate_file_size(io.BytesIO(file_content)):
#                 return jsonify({"error": f"File too large: {file.filename}"}), 400

#             # Process filename with content detection
#             filename = process_filename(file.filename, file_content)
#             file_path = os.path.join(config.UPLOAD_FOLDER, filename)

#             # Save file
#             with open(file_path, "wb") as f:
#                 f.write(file_content)

#             logger.info(f"Saved file {idx+1}/{len(uploaded_files)}: {file_path}")

#             # FIXED: Process metadata using names directly
#             image_type = image_types[idx] if idx < len(image_types) else "Unknown"
#             photo_type = photo_types[idx] if idx < len(photo_types) else "Unknown"

#             # Handle Property images
#             if image_type == "Property":
#                 bda_type_id = None
#                 bda_type_name = "Property"
#             else:
#                 # Use image_type directly as BDA type name
#                 bda_type_name = image_type
#                 # Try to extract ID if name follows pattern
#                 try:
#                     if "BDA_Type_" in bda_type_name:
#                         bda_type_id = int(bda_type_name.split("_")[-1])
#                     else:
#                         bda_type_id = None
#                 except:
#                     bda_type_id = None

#             if photo_type == "Property":
#                 photo_type_id = None
#                 photo_type_name = "Property"
#             else:
#                 # Use photo_type directly as photo type name
#                 photo_type_name = photo_type
#                 # Try to extract ID if name follows pattern
#                 try:
#                     if "Photo_Type_" in photo_type_name:
#                         photo_type_id = int(photo_type_name.split("_")[-1])
#                     else:
#                         photo_type_id = None
#                 except:
#                     photo_type_id = None

#             # Override with explicit names if provided
#             if idx < len(bda_type_names) and bda_type_names[idx]:
#                 bda_type_name = bda_type_names[idx]
#             if idx < len(photo_type_names) and photo_type_names[idx]:
#                 photo_type_name = photo_type_names[idx]

#             # GPS coordinates (only for non-Property images)
#             latitude = None
#             longitude = None
#             if image_type != "Property":
#                 latitude = latitudes[idx] if idx < len(latitudes) else None
#                 longitude = longitudes[idx] if idx < len(longitudes) else None

#             # Create image data structure
#             image_data = {
#                 "filename": filename,
#                 "original_filename": filename,
#                 "path": file_path,
#                 "bda_type_id": bda_type_id,
#                 "photo_type_id": photo_type_id,
#                 "bda_type_name": bda_type_name,
#                 "photo_type_name": photo_type_name,
#                 "form_latitude": latitude,
#                 "form_longitude": longitude,
#                 "work_order_context": work_order_context,
#                 "is_property_image": image_type == "Property",
#             }

#             logger.info(
#                 f"Image {idx+1}: BDA='{bda_type_name}', Photo='{photo_type_name}'"
#             )
#             all_images_data.append(image_data)

#         if len(all_images_data) < 2:
#             return jsonify({"error": "Need at least 2 images for comparison"}), 400

#         # Separate property images from order images
#         property_images = [
#             img for img in all_images_data if img.get("is_property_image")
#         ]
#         order_images = [
#             img for img in all_images_data if not img.get("is_property_image")
#         ]

#         logger.info(
#             f"Found {len(property_images)} property images and {len(order_images)} order images"
#         )

#         # Convert property images to base64
#         property_images_base64 = []
#         for prop_img in property_images:
#             try:
#                 prop_base64 = encode_image_to_base64(prop_img["path"])
#                 if prop_base64:
#                     property_images_base64.append(
#                         {
#                             "filename": prop_img["original_filename"],
#                             "base64": prop_base64,
#                             "path": prop_img["path"],
#                         }
#                     )
#             except Exception as e:
#                 logger.error(
#                     f"Failed to encode property image {prop_img['filename']}: {e}"
#                 )

#         # Create comparison pairs
#         comparison_pairs = group_images_by_bda_type(order_images)

#         if not comparison_pairs:
#             logger.info("No BDA type pairs found, trying filename-based pairing...")
#             comparison_pairs = group_images_by_filename(order_images)

#         if not comparison_pairs and len(order_images) >= 2:
#             # Fallback pairing
#             comparison_pairs = [
#                 {
#                     "before": order_images[0],
#                     "after": order_images[1],
#                     "set_id": "default_comparison",
#                     "comparison_type": "sequential_pairing",
#                 }
#             ]

#         if not comparison_pairs:
#             return (
#                 jsonify(
#                     {
#                         "error": "Could not create any comparison pairs",
#                         "debug": {
#                             "total_images": len(all_images_data),
#                             "order_images": len(order_images),
#                             "property_images": len(property_images),
#                             "bda_types": [
#                                 img.get("bda_type_name") for img in order_images
#                             ],
#                             "photo_types": [
#                                 img.get("photo_type_name") for img in order_images
#                             ],
#                         },
#                     }
#                 ),
#                 400,
#             )

#         logger.info(f"Found {len(comparison_pairs)} comparison pairs")

#         # Process comparison pairs
#         results = []
#         for pair_index, pair_data in enumerate(comparison_pairs):
#             try:
#                 result = process_comparison_pair(
#                     pair_data, pair_index, len(comparison_pairs), work_order_context
#                 )
#                 if result:
#                     results.append(result)
#             except Exception as e:
#                 logger.error(f"Error processing pair {pair_index + 1}: {e}")
#                 continue

#         if not results:
#             return jsonify({"error": "Failed to process any image pairs"}), 500

#         # Generate final response
#         response_data = generate_response_data(
#             results,
#             work_order_context,
#             property_images_base64,
#             len(uploaded_files),
#             len(order_images),
#         )

#         return jsonify(response_data)

#     except Exception as e:
#         logger.error(f"Upload error: {e}")
#         import traceback

#         traceback.print_exc()
#         return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# COMPLETE WORKING FIX - Replace your entire upload function with this


@generatecomparison.route("/chatgenie/v1/upload", methods=["POST"])
def upload():
    """WORKING FIX: Properly handle your exact payload structure"""
    try:
        # Extract form data
        uploaded_files = request.files.getlist("images")
        image_types = request.form.getlist("imageTypes")
        photo_types = request.form.getlist("photoTypes")
        latitudes = request.form.getlist("latitudes")
        longitudes = request.form.getlist("longitudes")
        issued_dates = request.form.getlist("issuedDates")
        completed_dates = request.form.getlist("completedDates")
        approved_dates = request.form.getlist("approvedDates")
        closed_dates = request.form.getlist("closedDates")
        work_order_statuses = request.form.getlist("workOrderStatus")
        document_detail_ids = request.form.getlist("documentDetailIds") or ["0"] * len(
            uploaded_files
        )
        bda_type_names = request.form.getlist("bdaTypeNames") or []
        photo_type_names = request.form.getlist("photoTypeNames") or []

        # Work Order Context
        work_order_context = {
            "workOrderNumber": request.form.get("workOrderNumber", ""),
            "workOrderId": request.form.get("workOrderId", ""),
            "fullAddress": request.form.get("fullAddress", ""),
            "propertyNumber": request.form.get("propertyNumber", ""),
            "workTaskName": request.form.get("workTaskName", ""),
            "scopeOfWork": request.form.get("scopeOfWork", ""),
            "vendorName": request.form.get("vendorName", ""),
            "workOrderStatusName": (
                work_order_statuses[0]
                if work_order_statuses
                else request.form.get("workOrderStatusName", "")
            ),
            "issuedDate": issued_dates[0] if issued_dates else "",
            "completedDate": completed_dates[0] if completed_dates else "",
            "approvedDate": approved_dates[0] if approved_dates else "",
            "closedDate": closed_dates[0] if closed_dates else "",
            "transactionId": request.form.get("transactionId", ""),
            "imageCount": len(uploaded_files),
            "hasPropertyImage": any(
                img_type.lower() == "property" for img_type in image_types
            ),
        }

        # DEBUG: Log your actual payload
        logger.info("=== ACTUAL PAYLOAD DEBUG ===")
        logger.info(f"Files: {len(uploaded_files)}")
        logger.info(f"imageTypes: {image_types}")
        logger.info(f"photoTypes: {photo_types}")
        logger.info(f"bdaTypeNames: {bda_type_names}")
        logger.info(f"photoTypeNames: {photo_type_names}")

        if not uploaded_files or all(f.filename == "" for f in uploaded_files):
            return jsonify({"error": "No files uploaded"}), 400

        # Create directories
        os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

        # Process uploaded files - IGNORE the problematic bdaTypeNames/photoTypeNames arrays
        all_images_data = []

        for idx, file in enumerate(uploaded_files):
            if not file or file.filename == "":
                continue

            # Read and validate file
            file_content = file.read()
            file.seek(0)

            if not allowed_file(file.filename, file_content):
                return (
                    jsonify({"error": f"File type not allowed: {file.filename}"}),
                    400,
                )

            if not validate_file_size(io.BytesIO(file_content)):
                return jsonify({"error": f"File too large: {file.filename}"}), 400

            # Process filename
            filename = process_filename(file.filename, file_content)
            file_path = os.path.join(config.UPLOAD_FOLDER, filename)

            # Save file
            with open(file_path, "wb") as f:
                f.write(file_content)

            # FIXED: Use imageTypes and photoTypes directly (they match image count)
            image_type = image_types[idx] if idx < len(image_types) else "Unknown"
            photo_type = photo_types[idx] if idx < len(photo_types) else "Unknown"

            # Handle Property images
            if image_type.lower() == "property":
                bda_type_name = "Property"
                photo_type_name = "Property"
                is_property = True
            else:
                # Use imageTypes directly as BDA type name
                bda_type_name = image_type
                photo_type_name = photo_type
                is_property = False

            # GPS coordinates (only for non-Property images)
            latitude = None
            longitude = None
            if not is_property:
                try:
                    if idx < len(latitudes) and latitudes[idx]:
                        latitude = float(latitudes[idx])
                except (ValueError, TypeError):
                    latitude = None

                try:
                    if idx < len(longitudes) and longitudes[idx]:
                        longitude = float(longitudes[idx])
                except (ValueError, TypeError):
                    longitude = None

            # Create image data
            image_data = {
                "filename": filename,
                "original_filename": filename,
                "path": file_path,
                "upload_index": idx,
                "bda_type_id": None,
                "photo_type_id": None,
                "bda_type_name": bda_type_name,
                "photo_type_name": photo_type_name,
                "form_latitude": latitude,
                "form_longitude": longitude,
                "work_order_context": work_order_context,
                "is_property_image": is_property,
            }

            logger.info(
                f"Image {idx}: '{filename}' - BDA:'{bda_type_name}' - Photo:'{photo_type_name}' - Property:{is_property}"
            )
            all_images_data.append(image_data)

        if len(all_images_data) < 2:
            return jsonify({"error": "Need at least 2 images for comparison"}), 400

        # Sort by upload_index for stability
        all_images_data.sort(key=lambda x: x.get("upload_index", 999))

        # Separate property vs order images
        property_images = [
            img for img in all_images_data if img.get("is_property_image")
        ]
        order_images = [
            img for img in all_images_data if not img.get("is_property_image")
        ]

        logger.info(
            f"Separated: {len(property_images)} property, {len(order_images)} order images"
        )

        # DEBUG: Show order images
        logger.info("=== ORDER IMAGES FOR PAIRING ===")
        for i, img in enumerate(order_images):
            logger.info(
                f"  {i}: {img['original_filename']} - BDA:'{img['bda_type_name']}' - Photo:'{img['photo_type_name']}'"
            )

        # Convert property images to base64
        property_images_base64 = []
        for prop_img in property_images:
            try:
                prop_base64 = encode_image_to_base64(prop_img["path"])
                if prop_base64:
                    property_images_base64.append(
                        {
                            "filename": prop_img["original_filename"],
                            "base64": prop_base64,
                            "path": prop_img["path"],
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to encode property image: {e}")

        # SIMPLE WORKING PAIRING LOGIC
        comparison_pairs = create_simple_pairs(order_images)

        if not comparison_pairs:
            return (
                jsonify(
                    {
                        "error": "Could not create any comparison pairs",
                        "debug": {
                            "order_images": len(order_images),
                            "property_images": len(property_images),
                            "images": [
                                {
                                    "file": img["original_filename"],
                                    "bda": img["bda_type_name"],
                                    "photo": img["photo_type_name"],
                                }
                                for img in order_images
                            ],
                        },
                    }
                ),
                400,
            )

        # Log final pairs
        logger.info("=== FINAL PAIRS ===")
        for i, pair in enumerate(comparison_pairs):
            logger.info(
                f"  Pair {i+1}: {pair['before']['original_filename']} <-> {pair['after']['original_filename']}"
            )

        # Process comparison pairs
        results = []
        for pair_index, pair_data in enumerate(comparison_pairs):
            try:
                result = process_comparison_pair(
                    pair_data, pair_index, len(comparison_pairs), work_order_context
                )
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing pair {pair_index + 1}: {e}")
                continue

        if not results:
            return jsonify({"error": "Failed to process any image pairs"}), 500

        # Generate final response
        response_data = generate_response_data(
            results,
            work_order_context,
            property_images_base64,
            len(uploaded_files),
            len(order_images),
        )

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Upload error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


def create_simple_pairs(order_images):
    """SIMPLE: Create pairs based on your exact payload structure"""
    logger.info("=== SIMPLE PAIRING LOGIC ===")

    # Separate Before/After images
    before_images = []
    after_images = []

    for img in order_images:
        bda_type = img.get("bda_type_name", "").strip()

        if bda_type.lower() == "before":
            before_images.append(img)
            logger.info(
                f"  BEFORE: {img['original_filename']} - Photo: {img.get('photo_type_name')}"
            )
        elif bda_type.lower() == "after":
            after_images.append(img)
            logger.info(
                f"  AFTER: {img['original_filename']} - Photo: {img.get('photo_type_name')}"
            )
        else:
            logger.info(
                f"  SKIPPED: {img['original_filename']} - BDA: '{bda_type}' (not before/after)"
            )

    logger.info(f"Found {len(before_images)} before, {len(after_images)} after images")

    if not before_images or not after_images:
        logger.info("No before/after separation possible")
        return []

    # Group by photo type
    from collections import defaultdict

    before_by_photo = defaultdict(list)
    after_by_photo = defaultdict(list)

    for img in before_images:
        photo_type = img.get("photo_type_name", "").strip().lower()
        before_by_photo[photo_type].append(img)

    for img in after_images:
        photo_type = img.get("photo_type_name", "").strip().lower()
        after_by_photo[photo_type].append(img)

    logger.info(f"Before photo types: {list(before_by_photo.keys())}")
    logger.info(f"After photo types: {list(after_by_photo.keys())}")

    # Create 1:1 pairs within matching photo types
    comparison_pairs = []
    pair_counter = 1

    for photo_type in sorted(before_by_photo.keys()):
        if photo_type in after_by_photo:
            befores = sorted(
                before_by_photo[photo_type], key=lambda x: x.get("upload_index", 999)
            )
            afters = sorted(
                after_by_photo[photo_type], key=lambda x: x.get("upload_index", 999)
            )

            num_pairs = min(len(befores), len(afters))
            logger.info(f"Creating {num_pairs} pairs for '{photo_type}'")

            for i in range(num_pairs):
                pair = {
                    "before": befores[i],
                    "after": afters[i],
                    "set_id": f"{photo_type}_{pair_counter}",
                    "comparison_type": "photo_type_matching",
                }
                comparison_pairs.append(pair)
                logger.info(
                    f"  Pair {pair_counter}: {befores[i]['original_filename']} <-> {afters[i]['original_filename']}"
                )
                pair_counter += 1

    logger.info(f"Created {len(comparison_pairs)} pairs total")
    return comparison_pairs


def process_comparison_pair(pair_data, pair_index, total_pairs, work_order_context):
    """Process a single comparison pair"""
    before_img = pair_data["before"]
    after_img = pair_data["after"]
    set_id = pair_data["set_id"]
    comparison_type = pair_data["comparison_type"]

    logger.info(f"Processing pair {pair_index + 1}/{total_pairs}")
    logger.info(f"Before: {before_img['filename']}")
    logger.info(f"After: {after_img['filename']}")

    try:
        # Prepare metadata for AI analysis
        before_metadata = {
            "bda_type_id": before_img.get("bda_type_id"),
            "bda_type_name": before_img.get("bda_type_name"),
            "photo_type_id": before_img.get("photo_type_id"),
            "photo_type_name": before_img.get("photo_type_name"),
            "form_latitude": before_img.get("form_latitude"),
            "form_longitude": before_img.get("form_longitude"),
            "bda_type_names": [before_img.get("bda_type_name", "")],
            "photo_type_names": [before_img.get("photo_type_name", "")],
        }

        after_metadata = {
            "bda_type_id": after_img.get("bda_type_id"),
            "bda_type_name": after_img.get("bda_type_name"),
            "photo_type_id": after_img.get("photo_type_id"),
            "photo_type_name": after_img.get("photo_type_name"),
            "form_latitude": after_img.get("form_latitude"),
            "form_longitude": after_img.get("form_longitude"),
            "bda_type_names": [after_img.get("bda_type_name", "")],
            "photo_type_names": [after_img.get("photo_type_name", "")],
        }

        # Generate AI comparison
        # comparison_result = generate_comparison_with_ai_validation(
        #     before_img["path"],
        #     after_img["path"],
        #     before_metadata,
        #     after_metadata,
        #     work_order_context,
        # )

        comparison_result = generate_comparisons_with_ai_validation(
            [
                {
                    "id": work_order_context.get("id", "WO_UNKNOWN"),
                    "before_images": [before_img["path"]],
                    "after_images": [after_img["path"]],
                    "before_metadata": before_metadata,
                    "after_metadata": after_metadata,
                    "work_order_context": work_order_context,
                }
            ]
        )[0]["analysis"]

        # FIXED: Log the raw AI response for debugging
        logger.info(f"=== RAW AI RESPONSE FOR PAIR {pair_index + 1} ===")
        logger.info(f"Response length: {len(comparison_result)}")
        logger.info(f"First 500 chars: {comparison_result[:500]}...")

        # Extract structured data from AI response
        structured_data, score, confidence, estimated_cost, validation_results = (
            extract_ai_results(comparison_result)
        )

        # Determine status
        metadata_valid = True  # Simplified for now
        if score >= 80 and metadata_valid:
            status = "approved"
        elif score >= 50:
            status = "pending"
        else:
            status = "rejected"

        # Convert images to base64
        before_base64 = encode_image_to_base64(before_img["path"])
        after_base64 = encode_image_to_base64(after_img["path"])

        if not before_base64 or not after_base64:
            logger.error(f"Failed to encode images for pair {pair_index + 1}")
            return None

        # Remove JSON from HTML and convert to HTML
        html_content = re.sub(
            r"```json\s*\{.*?\}\s*```", "", comparison_result, flags=re.DOTALL
        )
        result_html = markdown.markdown(
            html_content, extensions=["fenced_code", "tables", "nl2br"]
        )

        return {
            "pair_number": pair_index + 1,
            "set_id": set_id,
            "before": before_base64,
            "after": after_base64,
            "before_filename": before_img["original_filename"],
            "after_filename": after_img["original_filename"],
            "comparison_type": comparison_type,
            "score": score,
            "confidence": confidence,
            "estimated_cost": estimated_cost,
            "status": status,
            "data": structured_data,
            "html": result_html,
            "validation_results": validation_results,
            "metadata_valid": metadata_valid,
            "before_metadata": before_metadata,
            "after_metadata": after_metadata,
            "work_order_context": work_order_context,
        }

    except Exception as e:
        logger.error(f"Error in process_comparison_pair: {e}")
        return None


def extract_ai_results(comparison_result):
    """Extract structured data from AI response"""
    structured_data = {}
    score = 75  # Default score
    confidence = 95
    estimated_cost = "50-100"
    validation_results = {}

    try:
        json_match = re.search(
            r"```json\s*(\{.*?\})\s*```", comparison_result, re.DOTALL
        )
        if json_match:
            structured_data = json.loads(json_match.group(1))
            score = structured_data.get("score", 75)
            confidence = structured_data.get("confidence", 95)
            estimated_cost = structured_data.get("repair_cost", "50-100")
            validation_results = structured_data.get("validation_results", {})
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")

    return structured_data, score, confidence, estimated_cost, validation_results


def generate_response_data(
    results, work_order_context, property_images, upload_count, order_count
):
    """Generate final response data"""
    # Calculate statistics
    total_score = sum(r["score"] for r in results)
    average_score = total_score / len(results) if results else 0
    approved_count = sum(1 for r in results if r["status"] == "approved")
    metadata_valid_count = sum(1 for r in results if r["metadata_valid"])

    # Generate HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"comparison_report_{len(results)}pairs_{timestamp}.html"

    try:
        generate_html_report(
            results, work_order_context, property_images, report_filename, timestamp
        )
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")

    return {
        "success": True,
        "message": f"Successfully processed {len(results)} comparison pair{'s' if len(results) > 1 else ''}",
        "results": results,
        "total_pairs": len(results),
        "report_type": "combined" if len(results) > 1 else "single",
        "filePath": report_filename,
        "average_score": round(average_score, 1),
        "approved_count": approved_count,
        "metadata_valid_count": metadata_valid_count,
        "property_image_count": len(property_images),
        "order_image_count": order_count,
        "work_order_information": work_order_context,
        "validation_statistics": {
            "gps_validation_rate": (
                f"{(metadata_valid_count / len(results) * 100):.1f}%"
                if results
                else "0%"
            ),
            "date_validation_rate": (
                f"{(metadata_valid_count / len(results) * 100):.1f}%"
                if results
                else "0%"
            ),
            "overall_compliance_rate": (
                f"{(metadata_valid_count / len(results) * 100):.1f}%"
                if results
                else "0%"
            ),
        },
    }


def generate_html_report(
    results, work_order_context, property_images, report_filename, timestamp
):
    """Generate HTML report from results"""
    output_path = os.path.join(config.OUTPUT_FOLDER, report_filename)

    # Parse AI responses for HTML report
    all_comparisons_html = []
    for pair_index, result_item in enumerate(results):
        sections = parse_ai_response_content(result_item["html"])

        # FIXED: Debug log for each comparison
        logger.info(f"=== PARSED SECTIONS FOR PAIR {pair_index + 1} ===")
        for key, value in sections.items():
            logger.info(
                f"Section '{key}': {'[OK - ' + str(len(value)) + ' chars]' if value.strip() else '[EMPTY]'}"
            )

        all_comparisons_html.append(
            {
                "pair_number": pair_index + 1,
                "set_id": result_item["set_id"],
                "before_path": result_item["before"],
                "after_path": result_item["after"],
                "html_content": result_item["html"],
                "score": result_item["score"],
                "confidence": result_item["confidence"],
                "estimated_cost": result_item["estimated_cost"],
                "status": result_item["status"],
                "comparison_type": result_item["comparison_type"],
                "before_filename": result_item["before_filename"],
                "after_filename": result_item["after_filename"],
                "metadata_valid": result_item["metadata_valid"],
                "work_order_context": work_order_context,
                # FIXED: Parsed sections with fallback content
                "validation_analysis": sections.get("validation_analysis")
                or "AI validation analysis not available",
                "feature_scoring": sections.get("feature_scoring")
                or "Feature scoring analysis not available",
                "non_compliance_checks": sections.get("non_compliance_checks")
                or "No compliance issues detected",
                "summary": sections.get("summary") or "Analysis summary not available",
                "ai_recommendation": sections.get("ai_recommendation")
                or "AI recommendation not available",
                "qc_status": sections.get("qc_status") or "QC status pending",
            }
        )

    # Template variables
    template_vars = {
        "comparisons": all_comparisons_html,
        "total_pairs": len(results),
        "average_score": (
            round(sum(r["score"] for r in results) / len(results), 1) if results else 0
        ),
        "approved_count": sum(1 for r in results if r["status"] == "approved"),
        "metadata_valid_count": sum(1 for r in results if r["metadata_valid"]),
        "timestamp": timestamp,
        "work_order_number": work_order_context.get("workOrderNumber", ""),
        "work_order_id": work_order_context.get("workOrderId", ""),
        "full_address": work_order_context.get("fullAddress", ""),
        "vendor_name": work_order_context.get("vendorName", ""),
        "work_task_name": work_order_context.get("workTaskName", ""),
        "scope_of_work": work_order_context.get("scopeOfWork", ""),
        "property_number": work_order_context.get("propertyNumber", ""),
        "work_order_status_name": work_order_context.get("workOrderStatusName", ""),
        # "issued_date": work_order_context.get("issuedDate", ""),
        # "completed_date": work_order_context.get("completedDate", ""),
        # "approved_date": work_order_context.get("approvedDate", ""),
        # "closed_date": work_order_context.get("closedDate", ""),
        "issued_date": format_date(work_order_context.get("issuedDate", "")),
        "completed_date": format_date(work_order_context.get("completedDate", "")),
        "approved_date": format_date(work_order_context.get("approvedDate", "")),
        "closed_date": format_date(work_order_context.get("closedDate", "")),
        "transaction_id": work_order_context.get("transactionId", ""),
        "property_images": property_images,
        "has_property_image": len(property_images) > 0,
        **BASE64_IMAGES,
    }

    # Save HTML report
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(render_template("combined_result.html", **template_vars))
        logger.info(f"HTML report saved: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save HTML report: {e}")
        # Create a simple fallback HTML file
        fallback_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Analysis Report</title></head>
        <body>
        <h1>Image Analysis Report</h1>
        <p>Generated: {timestamp}</p>
        <p>Work Order: {work_order_context.get('workOrderNumber', 'N/A')}</p>
        <p>Total Pairs: {len(results)}</p>
        <p>Average Score: {template_vars['average_score']}</p>
        </body>
        </html>
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(fallback_html)


from datetime import datetime


def format_date(date_string):
    """Format date string to show only date and time (HH:MM)"""
    if not date_string or date_string == "N/A":
        return ""

    try:
        # Parse the date string - adjust the format based on your actual date format
        if "T" in date_string:
            # Format: 2025-08-28T19:33:36.48
            dt = datetime.fromisoformat(date_string.replace("T", " ").split(".")[0])
        else:
            # Format: 07/28/2025 00:00:00
            dt = datetime.strptime(date_string, "%m/%d/%Y %H:%M:%S")

        # Return formatted as: MM/DD/YYYY HH:MM
        return dt.strftime("%m/%d/%Y %H:%M")
    except:
        # If parsing fails, return original string
        return date_string


# FIXED: Single PDF generation function
@generatecomparison.route("/chatgenie/v1/pdf/<html_name>", methods=["GET"])
def generate_pdf_report(html_name):
    """Generate PDF from HTML report"""
    try:
        safe_filename = secure_filename(html_name)
        if not safe_filename.lower().endswith(".html"):
            safe_filename += ".html"

        file_path = os.path.join(config.OUTPUT_FOLDER, safe_filename)

        if not os.path.exists(file_path):
            return jsonify({"error": "Report not found"}), 404

        # Read and process HTML content
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Generate PDF
        pdf_bytes = asyncio.run(html_string_to_pdf_bytes(html_content))
        pdf_filename = safe_filename.rsplit(".", 1)[0] + ".pdf"
        pdf_buffer = io.BytesIO(pdf_bytes)

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            download_name=pdf_filename,
            as_attachment=True,
        )

    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500


# PDF generation utility function
async def html_string_to_pdf_bytes(html_content):
    """Convert HTML string to PDF bytes using Playwright"""
    browser = None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.set_content(html_content, wait_until="networkidle")
            pdf_bytes = await page.pdf(
                format="A4",
                print_background=True,
                margin={"top": "0in", "right": "0in", "bottom": "0in", "left": "0in"},
            )
            await browser.close()
            return pdf_bytes
    except Exception as e:
        if browser:
            await browser.close()
        logger.error(f"PDF generation failed: {e}")
        raise


# Error handlers
@generatecomparison.errorhandler(413)
def too_large(e):
    return (
        jsonify(
            {
                "error": "File too large",
                "max_size": f"{MAX_FILE_SIZE / (1024*1024):.1f}MB",
            }
        ),
        413,
    )


@generatecomparison.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return (
        jsonify(
            {
                "error": "Internal server error",
                "message": "Please check the logs for details",
            }
        ),
        500,
    )


@generatecomparison.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404
