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
import math  # Added for GPS distance calculations


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
    # If no extension, assume it's a JPEG (common for mobile uploads)
    if "." not in filename:
        return True  # Will be handled in secure_filename processing

    # Check if extension is in allowed list
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def process_filename(filename):
    """Process filename to ensure it has an extension"""
    secure_name = secure_filename(filename)

    # If no extension, add .jpg as default
    if "." not in secure_name:
        secure_name += ".jpg"

    return secure_name


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


def encode_image(image_path):
    """Encode image to base64"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        raise


def convert_gps_to_decimal(gps_info):
    """Convert GPS coordinates from DMS to decimal degrees"""
    try:

        def convert_coordinate(coord, ref):
            if not coord or len(coord) != 3:
                return None

            degrees = float(coord[0])
            minutes = float(coord[1])
            seconds = float(coord[2])

            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

            if ref in ["S", "W"]:
                decimal = -decimal

            return decimal

        lat = convert_coordinate(
            gps_info.get("GPSLatitude"), gps_info.get("GPSLatitudeRef")
        )
        lon = convert_coordinate(
            gps_info.get("GPSLongitude"), gps_info.get("GPSLongitudeRef")
        )

        return lat, lon
    except Exception as e:
        logger.error(f"GPS conversion error: {e}")
        return None, None


def get_image_metadata(image_path):
    """Extract and return comprehensive metadata from an image"""
    try:
        with Image.open(image_path) as img:
            # Use getexif() instead of deprecated _getexif()
            exif_data = img.getexif()

            if not exif_data:
                return {
                    "message": "No EXIF metadata found.",
                    "latitude": None,
                    "longitude": None,
                    "datetime_original": None,
                    "datetime_created": None,
                    "camera_info": {},
                    "gps_info": {},
                }

            metadata = {}
            gps_info = {}
            camera_info = {}

            # Process all EXIF tags
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                metadata[tag_name] = value

                # Extract camera information
                if tag_name in ["Make", "Model", "Software"]:
                    camera_info[tag_name.lower()] = str(value)

                # Extract GPS information
                if tag_name == "GPSInfo" and isinstance(value, dict):
                    for gps_tag_id, gps_value in value.items():
                        gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_info[gps_tag_name] = gps_value

            # Convert GPS to decimal coordinates
            latitude, longitude = (
                convert_gps_to_decimal(gps_info) if gps_info else (None, None)
            )

            # Extract and parse datetime
            datetime_original = None
            datetime_created = None

            if "DateTimeOriginal" in metadata:
                try:
                    datetime_original = {
                        "raw": str(metadata["DateTimeOriginal"]),
                        "formatted": datetime.strptime(
                            str(metadata["DateTimeOriginal"]), "%Y:%m:%d %H:%M:%S"
                        ).isoformat(),
                    }
                except ValueError:
                    datetime_original = {
                        "raw": str(metadata["DateTimeOriginal"]),
                        "formatted": None,
                    }

            if "DateTime" in metadata:
                try:
                    datetime_created = {
                        "raw": str(metadata["DateTime"]),
                        "formatted": datetime.strptime(
                            str(metadata["DateTime"]), "%Y:%m:%d %H:%M:%S"
                        ).isoformat(),
                    }
                except ValueError:
                    datetime_created = {
                        "raw": str(metadata["DateTime"]),
                        "formatted": None,
                    }

            return {
                "latitude": latitude,
                "longitude": longitude,
                "datetime_original": datetime_original,
                "datetime_created": datetime_created,
                "camera_info": camera_info,
                "gps_info": gps_info,
                "all_metadata": metadata,
            }

    except Exception as e:
        logger.error(f"Failed to extract metadata from {image_path}: {e}")
        return {
            "error": str(e),
            "latitude": None,
            "longitude": None,
            "datetime_original": None,
            "datetime_created": None,
            "camera_info": {},
            "gps_info": {},
        }


# =============================================================================
# AI-POWERED GPS + DATE VALIDATION FUNCTIONS
# =============================================================================


def validate_gps_coordinates(form_lat, form_lon, exif_lat, exif_lon, tolerance=0.001):
    """Simplified GPS validation - AI handles the real validation"""
    logger.info("GPS validation bypassed - AI handles validation")
    return True, {"message": "AI validation enabled - backend validation bypassed"}


def validate_date_range(image_datetime, issued_date, completed_date):
    """Simplified date validation - AI handles the real validation"""
    logger.info("Date validation bypassed - AI handles validation")
    return True, {"message": "AI validation enabled - backend validation bypassed"}


def validate_image_metadata(image_data, is_after_image=False):
    """Simplified metadata validation - AI handles the real validation"""
    validation_result = {
        "filename": image_data.get("filename", "unknown"),
        "is_after_image": is_after_image,
        "validations": {
            "gps": {"valid": True, "details": {"message": "AI validation enabled"}},
            "date": {"valid": True, "details": {"message": "AI validation enabled"}},
        },
        "violations": [],  # No violations - let AI decide
        "overall_valid": True,  # Always true - let AI decide
        "validation_summary": "✅ AI Validation Enabled",
    }

    # Update image_data with simplified validation results
    image_data["gps_coordinates_valid"] = True
    image_data["date_compliance_valid"] = True
    image_data["metadata_violations"] = []
    image_data["validation_details"] = validation_result

    logger.info(
        f"Simplified validation for {image_data.get('filename')} - AI will handle all validation"
    )

    return validation_result


def get_system_prompt():
    """Return the GPS + Date validation system prompt"""
    return """You are an expert AI assistant trained in real estate preservation, REO property management, and field service quality control. You validate work order photos by comparing "Before" and "After" images with GPS location and date verification.
def get_system_prompt():
    """Return the system prompt for image comparison with metadata validation"""
    return """You are an expert AI assistant trained in real estate preservation, REO property management, and field service quality control. You validate work order photos by comparing "Before" and "After" images with strict metadata verification.

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
- **METADATA VALIDATION**: GPS coordinates and timestamp compliance

⚙ Visual Analysis Process:
- Analyze structural elements, lighting, angles, and backgrounds
- Identify work type and transformation quality
- Detect any signs of image manipulation or duplication
- Evaluate safety and compliance standards
- **VERIFY GPS COORDINATES**: Form lat/long must match image EXIF GPS data for both images
- **VERIFY TIMESTAMPS**: ONLY the AFTER image datetime must be between issued_date and completed_date

🚨 CRITICAL METADATA REQUIREMENTS:
1. **GPS Validation**: The provided form coordinates (latitude/longitude) MUST match the EXIF GPS data extracted from BOTH images. Any mismatch indicates potential fraud or incorrect documentation.
2. **Date Validation**: ONLY the AFTER image EXIF datetime_original MUST fall between the work order issued_date and completed_date. BEFORE images can be taken anytime before work starts.
3. **Location Consistency**: Both before and after images must have matching GPS coordinates to confirm same work location.

Follow this EXACT output format:

---

🖼 Title: [Brief description of work performed]

🔍 Validation Analysis

| Factor | Analysis | Result |
|--------|----------|--------|
| **GPS Validation** | **Form GPS coordinate consistency check** | **✅ / ❌** |
| **Date Compliance** | **Work period timeline validation** | **✅ / ❌** |
| Location Consistency | Same Form GPS coordinates for both images | ✅ / ❌ |
| **GPS Coordinates** | **Form lat/long vs EXIF GPS match (both images)** | **✅ / ❌** |
| **Date Compliance** | **AFTER image datetime within work period** | **✅ / ❌** |
| Location Match | GPS/metadata alignment assessment | ✅ / ❌ |
| Duplicate Check | Images identical or nearly same | ✅ / ❌ |
| Tampering Check | Evidence of editing or manipulation | ✅ / ❌ |
| Area Consistency | Same room/location verification | ✅ / ❌ |
| Work Scope | Expected work type visible | ✅ / ❌ |
| Photo Quality | Clarity, lighting, focus assessment | ✅ / ❌ |

📍 **Metadata Verification Details**

| Validation Type | Before Image | After Image | Status |
|----------------|--------------|-------------|---------|
| **Form GPS** | [lat, long from form] | [lat, long from form] | ✅ / ❌ |
| **EXIF GPS** | [lat, long from EXIF] | [lat, long from EXIF] | ✅ / ❌ |
| **GPS Match** | Form vs EXIF comparison | Form vs EXIF comparison | ✅ / ❌ |
| **Image DateTime** | [EXIF datetime] | [EXIF datetime] | ✅ / ❌ |
| **Work Period** | [Not validated - can be anytime] | [issued_date to completed_date] | ✅ / ❌ |
| **Date Compliance** | Not required for BEFORE | Within work period? | ✅ / ❌ |

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
- ✅ Approved (≥80 + GPS valid + date compliant) / ⚠ Review (50-79 or validation issues) / ❌ Rejected (<50 or validation failures)

⚠️ **VALIDATION VIOLATIONS**: 
[List GPS coordinate mismatches, timeline issues, or missing data]

📄 Summary:
[3-5 sentences describing changes, quality, concerns, GPS validation, and timeline compliance]
- ✅ Approved (≥80 + metadata valid) / ⚠ Review (50-79 or metadata issues) / ❌ Rejected (<50 or metadata invalid)

⚠️ **METADATA VIOLATIONS**: 
[List any GPS coordinate mismatches for either image or AFTER image date compliance failures - these are CRITICAL ISSUES]

📄 Summary:
[3-5 sentences describing changes, quality, concerns, and metadata validation status. Note that BEFORE images don't require date validation.]

**For backend processing, include JSON data:**

```json
{
  "score": [0-100],
  "verdict": "[status with emoji]",
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
  "metadata_valid": {
    "gps_coordinates_match": true/false,
    "after_date_compliance": true/false,
    "before_gps_valid": true/false,
    "after_gps_valid": true/false,
    "before_date_valid": true,
    "after_date_valid": true/false
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

**GPS + DATE VALIDATION INSTRUCTIONS:**
- GPS VALIDATION: Use Form GPS coordinates as authoritative - ensure both images have same coordinates
- DATE VALIDATION: Analyze work timeline using issued_date and completed_date for logical compliance
- LOCATION CONSISTENCY: Both images must have identical Form GPS coordinates
- TIMELINE LOGIC: Work should follow logical before/during/after sequence
- NO EXIF PARSING: Use form data only - no complex metadata extraction needed
- PENALTY LOGIC: Reduce score by 15 points for GPS violations, 15 points for date violations
- SIMPLE APPROACH: Focus on practical validation using available form data
- DUPLICATE CHECK: Do not check for duplicate or similar images - focus on work completion validation only"""


def create_ai_metadata_context(before_metadata, after_metadata):
    """
    GPS + Date metadata context for location and timeline validation.
    """
    context = "\n\n📍 GPS + DATE VALIDATION CONTEXT:\n"
    context += "=" * 80 + "\n"
    context += "🤖 INSTRUCTIONS: Validate GPS coordinates AND work timeline using form data only.\n\n"

    # GPS COORDINATES
    form_lat_before = before_metadata.get("form_latitude", "N/A")
    form_lon_before = before_metadata.get("form_longitude", "N/A")
    form_lat_after = after_metadata.get("form_latitude", "N/A")
    form_lon_after = after_metadata.get("form_longitude", "N/A")

    # WORK TIMELINE DATES
    issued_date = before_metadata.get("issued_date", "N/A")
    completed_date = before_metadata.get("completed_date", "N/A")

    context += f"🎯 DUAL VALIDATION REQUIREMENTS:\n"
    context += f"• GPS Validation: Coordinate consistency between images\n"
    context += f"• Date Validation: Timeline compliance with work order period\n"
    context += f"• Data Source: Form data only (no EXIF extraction needed)\n\n"

    # GPS VALIDATION SECTION
    context += f"🗺️ GPS COORDINATE VALIDATION:\n"
    context += f"• BEFORE Image GPS: {form_lat_before}, {form_lon_before}\n"
    context += f"• AFTER Image GPS: {form_lat_after}, {form_lon_after}\n"
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

    # DATE/TIMELINE VALIDATION SECTION
    context += f"⏰ WORK TIMELINE VALIDATION:\n"
    context += f"• Work Order Issued: {issued_date}\n"
    context += f"• Work Order Completed: {completed_date}\n"
    context += f"• Timeline Status: {'✅ Dates Available' if issued_date != 'N/A' and completed_date != 'N/A' else '❌ Missing Dates'}\n"

    if issued_date != "N/A" and completed_date != "N/A":
        context += f"• Work Period: {issued_date} to {completed_date}\n"
        context += f"• Timeline Logic Required: BEFORE taken before/during work, AFTER taken during/after work\n"
    else:
        context += f"• Timeline Validation: ❌ CANNOT VALIDATE (Missing work dates)\n"

    context += f"\n"

    # VALIDATION STRATEGY
    context += f"🎯 VALIDATION STRATEGY:\n"
    context += f"• GPS Validation:\n"
    context += f"  - Both images must have identical Form GPS coordinates\n"
    context += f"  - Exact coordinate matching required\n"
    context += f"  - Missing GPS = validation failure\n"
    context += f"• Date Validation:\n"
    context += f"  - Verify work timeline makes logical sense\n"
    context += f"  - BEFORE: Should be taken before or during early work phase\n"
    context += f"  - AFTER: Should be taken during or after work completion\n"
    context += f"  - Use logical reasoning based on work order dates\n"
    context += f"• Combined Assessment:\n"
    context += f"  - Both GPS and timeline must be valid for full approval\n"
    context += f"  - Partial compliance may warrant review status\n\n"

    # VALIDATION EXAMPLES
    context += f"📐 VALIDATION EXAMPLES:\n"
    context += f"• ✅ VALID GPS:\n"
    context += f"  - BEFORE GPS: (9.91976, 78.09494)\n"
    context += f"  - AFTER GPS: (9.91976, 78.09494)\n"
    context += f"  - Result: Same work site confirmed\n"
    context += f"• ✅ VALID TIMELINE:\n"
    context += f"  - Work Period: 2025-01-15 to 2025-01-30\n"
    context += f"  - Logic: 15-day work period appears reasonable\n"
    context += f"  - Timeline: Before/after sequence makes sense\n"
    context += f"• ❌ INVALID SCENARIOS:\n"
    context += f"  - Different GPS coordinates between images\n"
    context += f"  - Illogical work timeline (completed before issued)\n"
    context += f"  - Missing GPS or date information\n\n"

    # IMPLEMENTATION INSTRUCTIONS
    context += f"🔧 AI IMPLEMENTATION:\n"
    context += f"• Use form data only - no EXIF metadata extraction\n"
    context += f"• Apply logical reasoning for timeline validation\n"
    context += f"• Exact GPS coordinate matching required\n"
    context += f"• Consider work order timeline reasonableness\n"
    context += f"• Apply penalties: 15 points for GPS violations, 15 points for date violations\n"
    context += f"• Provide clear explanations for both validations\n"
    context += f"• Focus on practical compliance using available data\n"
    context += "=" * 80 + "\n"
**IMPORTANT**: If GPS coordinates don't match between form data and EXIF data for either image, OR if the AFTER image timestamp falls outside the work order period, automatically reduce the final score by 30 points and mark as "Review" or "Rejected" status regardless of visual quality. BEFORE images do NOT require date validation."""


def create_metadata_context(before_metadata, after_metadata):
    """Create context string with metadata for AI analysis"""
    context = "\n\n📍 METADATA CONTEXT FOR VALIDATION:\n"

    # Before image metadata
    context += f"BEFORE IMAGE METADATA:\n"
    context += f"- Form GPS: {before_metadata.get('form_latitude', 'N/A')}, {before_metadata.get('form_longitude', 'N/A')}\n"
    context += f"- EXIF GPS: {before_metadata.get('exif_latitude', 'N/A')}, {before_metadata.get('exif_longitude', 'N/A')}\n"
    context += f"- EXIF DateTime: {before_metadata.get('exif_datetime', {}).get('formatted', 'N/A') if before_metadata.get('exif_datetime') else 'N/A'}\n"
    context += (
        f"- Date Validation: NOT REQUIRED (BEFORE images can be taken anytime)\n\n"
    )

    # After image metadata
    context += f"AFTER IMAGE METADATA:\n"
    context += f"- Form GPS: {after_metadata.get('form_latitude', 'N/A')}, {after_metadata.get('form_longitude', 'N/A')}\n"
    context += f"- EXIF GPS: {after_metadata.get('exif_latitude', 'N/A')}, {after_metadata.get('exif_longitude', 'N/A')}\n"
    context += f"- EXIF DateTime: {after_metadata.get('exif_datetime', {}).get('formatted', 'N/A') if after_metadata.get('exif_datetime') else 'N/A'}\n"
    context += f"- Work Issued Date: {after_metadata.get('issued_date', 'N/A')}\n"
    context += f"- Work Completed Date: {after_metadata.get('completed_date', 'N/A')}\n"
    context += f"- Date Validation: REQUIRED (must be within work period)\n\n"

    context += "⚠️ VALIDATION REQUIREMENTS:\n"
    context += "1. GPS coordinates from form data MUST match EXIF GPS data for BOTH images (within ~50m tolerance)\n"
    context += "2. ONLY the AFTER image datetime MUST be between issued_date and completed_date\n"
    context += "3. BEFORE images do NOT require date validation (can be taken anytime before work)\n"
    context += (
        "4. Both images MUST have consistent GPS coordinates to confirm same location\n"
    )
    context += "5. Metadata violations result in automatic score reduction and review/rejection status\n"

    return context


def create_metadata_context(before_metadata, after_metadata, backend_validation=None):
    """
    Wrapper function for compatibility - now uses AI validation context.
    The backend_validation parameter is ignored.
    """
    return create_ai_metadata_context(before_metadata, after_metadata)


def generate_comparison_with_ai_validation(
    before_path, after_path, before_metadata=None, after_metadata=None
):
    """Generate comparison analysis with AI-based GPS + Date validation"""
def generate_comparison(
    before_path, after_path, before_metadata=None, after_metadata=None
):
    """Generate comparison analysis between before and after images with metadata validation"""
    try:
        before_data = encode_image(before_path)
        after_data = encode_image(after_path)

        # Create AI validation context
        metadata_context = ""
        if before_metadata and after_metadata:
            metadata_context = create_ai_metadata_context(
                before_metadata, after_metadata
            )
        # Create metadata context for AI
        metadata_context = ""
        if before_metadata and after_metadata:
            metadata_context = create_metadata_context(before_metadata, after_metadata)

        messages = [
            {"role": "system", "content": get_system_prompt()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Compare these before and after images. Analyze GPS coordinates and work timeline validation. "
                        f"Use logical reasoning to validate location consistency and timeline compliance.{metadata_context}",
                        "text": f"Compare these before and after images. Provide detailed analysis with scoring and verdict. Pay special attention to metadata validation.{metadata_context}",
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
            max_tokens=3000,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Failed to generate AI comparison: {e}")
        return f"Error generating comparison: {str(e)}"


def generate_comparison(
    before_path,
    after_path,
    before_metadata=None,
    after_metadata=None,
    backend_validation=None,
):
    """
    AI-powered comparison function - backend_validation parameter ignored for compatibility.
    This replaces the old backend validation approach.
    """
    return generate_comparison_with_ai_validation(
        before_path, after_path, before_metadata, after_metadata
    )


# =============================================================================
# MAIN APPLICATION ROUTES
# =============================================================================


@generatecomparison.route("/")
def index():
    """Render upload page"""
    return render_template("upload.html")


def group_images_by_filename(images_data):
    """
    Group images into sets based on the file naming convention:
    Before_<type>_image_<number>.<ext> and After_<type>_image_<number>.<ext>

    Examples:
    - Before_Exterior_image_0.jpg -> After_Exterior_image_1.jpg
    - Before_RoofTopPhoto_image_2.jpg -> After_RoofTopPhoto_image_3.jpg

    Returns a list of dicts with paired 'before' and 'after' images of the same type.
    """
    before_images = []
    after_images = []

    # Separate before and after images (case-insensitive)
    Group images into sets based on file naming convention for your specific format:
    before_<type>_image_<number>.<ext> and after_<type>_image_<number>.<ext>

    Examples:
    - before_exterior_image_0.jpg <-> after_exterior_image_1.jpg
    - before_interior_image_2.jpg <-> after_interior_image_3.jpg
    """
    import re

    before_images = []
    after_images = []

    # Separate before and after images
    for img_data in images_data:
        filename_lower = img_data["filename"].lower()
        if filename_lower.startswith("before_"):
            before_images.append(img_data)
        elif filename_lower.startswith("after_"):
            after_images.append(img_data)

    def extract_image_type(filename):
        """Extract the type from filename like Before_Exterior_image_0.jpg -> exterior"""
        filename_lower = filename.lower()

        # Remove 'before_' or 'after_' prefix (case-insensitive)
        """
        Extract the image type from filename
        before_exterior_image_0.jpg -> 'exterior'
        after_interior_image_1.jpg -> 'interior'
        """
        filename_lower = filename.lower()

        # Remove before_/after_ prefix
        if filename_lower.startswith("before_"):
            base = filename_lower[len("before_") :]
        elif filename_lower.startswith("after_"):
            base = filename_lower[len("after_") :]
        else:
            return None

        # Remove file extension
        base = base.rsplit(".", 1)[0] if "." in base else base

        # Handle different patterns:
        # Pattern 1: type_image_number (e.g., exterior_image_0)
        # Pattern 2: type_img (e.g., exterior_img)

        # Remove _image_<number> or _img suffix
        if "_image_" in base:
            # Extract everything before '_image_<number>'
            base = base.split("_image_")[0]
        elif base.endswith("_img"):
            # Remove '_img' suffix
            base = base[:-4]

        return base

    # Group images by type
    before_by_type = {}
    after_by_type = {}
        # Extract the type part (everything before '_image_')
        # Pattern: <type>_image_<number>.<ext>
        match = re.match(r"^(.+?)_image_\d+\.", base)
        if match:
            return match.group(1)  # Returns 'exterior', 'interior', etc.

        # Fallback: remove _image_number pattern
        type_part = re.sub(r"_image_\d+.*$", "", base)
        return type_part if type_part else base

    # Group by image types
    before_by_type = defaultdict(list)
    after_by_type = defaultdict(list)

    for img in before_images:
        img_type = extract_image_type(img["filename"])
        if img_type:
            before_by_type[img_type] = img
            logger.info(f"Found BEFORE image type '{img_type}': {img['filename']}")
            before_by_type[img_type].append(img)
            logger.info(f"Before image: {img['filename']} -> type: '{img_type}'")

    for img in after_images:
        img_type = extract_image_type(img["filename"])
        if img_type:
            after_by_type[img_type] = img
            logger.info(f"Found AFTER image type '{img_type}': {img['filename']}")

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
                f"✅ Created pair for '{img_type}': {before_by_type[img_type]['filename']} -> {after_by_type[img_type]['filename']}"
            )
        else:
            logger.warning(
                f"⚠️ No matching AFTER image found for BEFORE type '{img_type}'"
            )

    # Check for unmatched after images
    for img_type in after_by_type:
        if img_type not in before_by_type:
            logger.warning(
                f"⚠️ No matching BEFORE image found for AFTER type '{img_type}'"
            )

    logger.info(
        f"📊 Created {len(comparison_pairs)} comparison pairs from {len(before_images)} before and {len(after_images)} after images"
    )
    return comparison_pairs


@generatecomparison.route("/chatgenie/v1/upload", methods=["POST"])
def upload():
    """Handle image upload and comparison processing with AI-based GPS + Date validation"""
    try:
        uploaded_files = request.files.getlist("images")
            after_by_type[img_type].append(img)
            logger.info(f"After image: {img['filename']} -> type: '{img_type}'")

    # Create comparison pairs
    comparison_pairs = []

    for img_type in before_by_type:
        if img_type in after_by_type:
            before_list = before_by_type[img_type]
            after_list = after_by_type[img_type]

            # Sort by the number in filename for consistent pairing
            def get_image_number(img):
                filename = img["filename"].lower()
                match = re.search(r"image_(\d+)", filename)
                return int(match.group(1)) if match else 0

            before_list.sort(key=get_image_number)
            after_list.sort(key=get_image_number)

            # Pair them up (can handle multiple pairs of same type)
            for i in range(min(len(before_list), len(after_list))):
                before_img = before_list[i]
                after_img = after_list[i]

                comparison_pairs.append(
                    {
                        "before": before_img,
                        "after": after_img,
                        "set_id": f"{img_type}_{i+1}",
                        "comparison_type": "type_based_matching",
                    }
                )

                logger.info(
                    f"Paired: {before_img['filename']} <-> {after_img['filename']} (type: {img_type})"
                )

    return comparison_pairs


def validate_gps_coordinates(form_lat, form_lon, exif_lat, exif_lon, tolerance=0.001):
    """Validate if GPS coordinates match within tolerance (approximately 100 meters)"""
    try:
        if not all([form_lat, form_lon, exif_lat, exif_lon]):
            return False

        form_lat = float(form_lat)
        form_lon = float(form_lon)

        lat_diff = abs(form_lat - exif_lat)
        lon_diff = abs(form_lon - exif_lon)

        return lat_diff <= tolerance and lon_diff <= tolerance
    except (ValueError, TypeError):
        return False


def validate_date_range(image_datetime, issued_date, completed_date):
    """Validate if image datetime falls within work order date range"""
    try:
        if not all([image_datetime, issued_date, completed_date]):
            return False

        # Parse datetime string if it's in ISO format
        if isinstance(image_datetime, dict) and "formatted" in image_datetime:
            image_dt = datetime.fromisoformat(image_datetime["formatted"])
        else:
            image_dt = datetime.fromisoformat(str(image_datetime))

        issued_dt = datetime.fromisoformat(issued_date)
        completed_dt = datetime.fromisoformat(completed_date)

        return issued_dt <= image_dt <= completed_dt
    except (ValueError, TypeError) as e:
        logger.error(f"Date validation error: {e}")
        return False


@generatecomparison.route("/chatgenie/v1/upload", methods=["POST"])
def upload():
    """Handle image upload and comparison processing with enhanced metadata validation"""
    try:
        uploaded_files = request.files.getlist("images")

        image_types = request.form.getlist("imageTypes")  # BDATypeId values
        photo_types = request.form.getlist("photoTypes")  # PhotoTypeId values
        latitudes = request.form.getlist("latitudes")
        longitudes = request.form.getlist("longitudes")
        issued_dates = request.form.getlist("issuedDates")
        completed_dates = request.form.getlist("completedDates")

        # Debug logging
        logger.info("=== UPLOAD DEBUG INFO ===")
        logger.info(f"Image Types: {image_types}")
        logger.info(f"Photo Types: {photo_types}")
        logger.info(f"Latitudes: {latitudes}")
        logger.info(f"Longitudes: {longitudes}")
        logger.info(f"Issued Dates: {issued_dates}")
        logger.info(f"Completed Dates: {completed_dates}")
        logger.info(f"Uploaded Files: {[f.filename for f in uploaded_files]}")
        print("Image Types:", image_types)
        print("Photo Types:", photo_types)
        print("Latitudes:", latitudes)
        print("Longitudes:", longitudes)
        print("Issued Dates:", issued_dates)
        print("Completed Dates:", completed_dates)

        if not uploaded_files or all(f.filename == "" for f in uploaded_files):
            return jsonify({"error": "No files uploaded"}), 400

        logger.info(f"Received {len(uploaded_files)} files for processing")

        # Create directories
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Process and collect all images with metadata
        all_images_data = []

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

            # Process filename to ensure it has an extension
            filename = process_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            logger.info(f"Saved file {idx+1}/{len(uploaded_files)}: {file_path}")

            # Get form metadata for this file (with bounds checking)
            # Extract metadata from image
            image_metadata = get_image_metadata(file_path)
            logger.info(
                f"Extracted metadata for {filename}: GPS({image_metadata.get('latitude')}, {image_metadata.get('longitude')}), DateTime: {image_metadata.get('datetime_original', {}).get('formatted') if image_metadata.get('datetime_original') else 'None'}"
            )

            # Get metadata for this file
            bda_type_id = int(image_types[idx]) if idx < len(image_types) else 1
            photo_type_id = int(photo_types[idx]) if idx < len(photo_types) else 0
            latitude = latitudes[idx] if idx < len(latitudes) else None
            longitude = longitudes[idx] if idx < len(longitudes) else None
            issued_date = issued_dates[idx] if idx < len(issued_dates) else None
            completed_date = (
                completed_dates[idx] if idx < len(completed_dates) else None
            )

            logger.info(f"Form metadata for {filename}:")
            logger.info(f"  - Form GPS: lat={latitude}, lon={longitude}")
            logger.info(f"  - Work Period: {issued_date} to {completed_date}")

            # Create simplified image data structure (GPS + Date only)

            # Validate metadata immediately
            gps_valid = validate_gps_coordinates(
                latitude,
                longitude,
                image_metadata.get("latitude"),
                image_metadata.get("longitude"),
            )

            # Only validate date range for AFTER images (work completion photos)
            # BEFORE images can be taken anytime before work starts
            is_after_image = filename.lower().startswith("after_")
            date_valid = True  # Default to valid for BEFORE images

            if is_after_image:
                date_valid = validate_date_range(
                    image_metadata.get("datetime_original"), issued_date, completed_date
                )
            else:
                logger.info(f"Skipping date validation for BEFORE image: {filename}")

            image_data = {
                "filename": filename,
                "original_filename": filename,
                "path": file_path,
                "bda_type_id": bda_type_id,
                "photo_type_id": photo_type_id,
                "latitude": latitude,  # Form GPS only
                "longitude": longitude,  # Form GPS only
                "issued_date": issued_date,  # Work order dates
                "completed_date": completed_date,  # Work order dates
                "latitude": latitude,
                "longitude": longitude,
                "issued_date": issued_date,
                "completed_date": completed_date,
                "index": idx,
                # Add extracted EXIF metadata
                "exif_metadata": image_metadata,
                "exif_latitude": image_metadata.get("latitude"),
                "exif_longitude": image_metadata.get("longitude"),
                "exif_datetime_original": image_metadata.get("datetime_original"),
                "exif_datetime_created": image_metadata.get("datetime_created"),
                "exif_camera_info": image_metadata.get("camera_info", {}),
                "exif_gps_info": image_metadata.get("gps_info", {}),
                # Add validation results
                "gps_coordinates_valid": gps_valid,
                "date_compliance_valid": date_valid,
                "metadata_violations": [],
            }

            all_images_data.append(image_data)
            # Log validation results
            if not gps_valid:
                violation = f"GPS coordinates mismatch for {filename}: Form({latitude}, {longitude}) vs EXIF({image_metadata.get('latitude')}, {image_metadata.get('longitude')})"
                image_data["metadata_violations"].append(violation)
                logger.warning(violation)

            if not date_valid and is_after_image:
                violation = f"Date compliance failure for AFTER image {filename}: Image datetime outside work period ({issued_date} to {completed_date})"
                image_data["metadata_violations"].append(violation)
                logger.warning(violation)
            elif is_after_image and date_valid:
                logger.info(f"✅ Date validation passed for AFTER image {filename}")
            else:
                logger.info(f"ℹ️ Date validation skipped for BEFORE image {filename}")

            all_images_data.append(image_data)
            logger.info(f"Added {filename} with metadata validation results")

        if len(all_images_data) < 2:
            return jsonify({"error": "Need at least 2 images for comparison"}), 400

        # Use the improved filename-based pairing function
        comparison_pairs = group_images_by_filename(all_images_data)

        if not comparison_pairs:
            # Return error with debug info if no pairs found
        # Use filename-based pairing instead of complex grouping
        logger.info("Grouping images by filename convention (before_/after_)...")
        comparison_pairs = group_images_by_filename(all_images_data)

        if not comparison_pairs:
            # Log debug information about why no pairs were found
            before_files = [
                img["filename"]
                for img in all_images_data
                if img["filename"].lower().startswith("before_")
            ]
            after_files = [
                img["filename"]
                for img in all_images_data
                if img["filename"].lower().startswith("after_")
            ]

            return (
                jsonify(
                    {
                        "error": "No valid comparison pairs found. Ensure images are named with 'before_' and 'after_' prefixes and have matching types.",
                        "error": "No valid comparison pairs found. Check that images follow naming convention: before_<name>.<ext> and after_<name>.<ext>",
                        "debug": {
                            "total_images": len(all_images_data),
                            "before_files": before_files,
                            "after_files": after_files,
                            "naming_help": "Expected format: before_exterior_img.jpg, after_exterior_img.jpg",
                            "naming_help": "Expected format: before_kitchen.jpg, after_kitchen.jpg",
                        },
                    }
                ),
                400,
            )

        logger.info(
            f"Found {len(comparison_pairs)} comparison pairs based on filename matching"
            f"Found {len(comparison_pairs)} comparison pairs based on filenames"
        )

        results = []
        all_comparisons_html = []

        # Process each comparison pair with AI GPS + Date validation
        for pair_index, pair_data in enumerate(comparison_pairs):
            before_img = pair_data["before"]
            after_img = pair_data["after"]
            set_id = pair_data["set_id"]
            comparison_type = pair_data["comparison_type"]

            logger.info(
                f"=== PROCESSING PAIR {pair_index + 1}/{len(comparison_pairs)} ==="
            )
            logger.info(f"Set: '{set_id}' ({comparison_type})")
            logger.info(f"Before: {before_img['filename']}")
            logger.info(f"After: {after_img['filename']}")

            try:
                # Prepare GPS + Date metadata for AI analysis
                before_metadata = {
                # Prepare metadata for AI analysis
                before_metadata = {
                    "exif_latitude": before_img.get("exif_latitude"),
                    "exif_longitude": before_img.get("exif_longitude"),
                    "exif_datetime": before_img.get("exif_datetime_original"),
                    "exif_camera": before_img.get("exif_camera_info"),
                    "form_latitude": before_img.get("latitude"),
                    "form_longitude": before_img.get("longitude"),
                    "issued_date": before_img.get("issued_date"),
                    "completed_date": before_img.get("completed_date"),
                }

                after_metadata = {
                    "exif_latitude": after_img.get("exif_latitude"),
                    "exif_longitude": after_img.get("exif_longitude"),
                    "exif_datetime": after_img.get("exif_datetime_original"),
                    "exif_camera": after_img.get("exif_camera_info"),
                    "form_latitude": after_img.get("latitude"),
                    "form_longitude": after_img.get("longitude"),
                    "issued_date": after_img.get("issued_date"),
                    "completed_date": after_img.get("completed_date"),
                }

                # Generate comparison analysis with AI GPS + Date validation
                comparison_result = generate_comparison_with_ai_validation(
                # Generate comparison analysis with metadata
                comparison_result = generate_comparison(
                    before_img["path"],
                    after_img["path"],
                    before_metadata,
                    after_metadata,
                )

                # Extract score and structured data from AI response
                structured_data = {}
                score = 0
                validation_results = {}

                try:
                    json_match = re.search(
                        r"```json\s*(\{.*?\})\s*```", comparison_result, re.DOTALL
                    )
                    if json_match:
                        structured_data = json.loads(json_match.group(1))
                        score = structured_data.get("score", 0)
                        validation_results = structured_data.get(
                            "validation_results", {}
                        )
                    else:
                        logger.warning("No JSON object found in AI response")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")

                # Fallback: extract score from text if JSON not found
                if not score:
                    score_match = re.search(
                        r"score\s*[:=]\s*(\d+)", comparison_result, re.IGNORECASE
                    )
                    if score_match:
                        score = int(score_match.group(1))

                # Extract GPS and Date validation results
                gps_valid = validation_results.get("gps_coordinates_valid", True)
                date_valid = validation_results.get("date_compliance_valid", True)
                timeline_valid = validation_results.get("timeline_valid", True)
                location_consistent = validation_results.get(
                    "location_consistent", True
                )

                # Overall metadata validity (both GPS and date must be valid)
                metadata_valid = (
                    gps_valid and date_valid and location_consistent and timeline_valid
                )

                # Remove JSON block from the HTML output
                html_content = re.sub(
                    r"```json\s*\{.*?\}\s*```", "", comparison_result, flags=re.DOTALL
                )
                html_content = re.sub(
                    r"\*\*For backend processing, include JSON data.*?\*\*\s*",
                    "",
                    html_content,
                )

                # Convert markdown to HTML
                result_html = markdown.markdown(
                    html_content, extensions=["fenced_code", "tables", "nl2br"]
                )

                # Determine status based on score and validation results
                if score >= 80 and metadata_valid:
                    status = "approved"
                elif score >= 50 or not metadata_valid:
                # Apply metadata validation penalties
                metadata_violations = before_img.get(
                    "metadata_violations", []
                ) + after_img.get("metadata_violations", [])
                if metadata_violations:
                    logger.warning(
                        f"Metadata violations detected for pair {pair_index + 1}: {metadata_violations}"
                    )
                    score = max(
                        0, score - 30
                    )  # Reduce score by 30 points for metadata violations

                # Convert markdown to HTML
                result_html = markdown.markdown(
                    comparison_result, extensions=["fenced_code", "tables", "nl2br"]
                )

                # Determine status based on score and metadata validation
                has_metadata_violations = bool(metadata_violations)

                if score >= 80 and not has_metadata_violations:
                    status = "approved"
                elif score >= 50 or has_metadata_violations:
                    status = "pending"
                else:
                    status = "rejected"

                # Updated logging to include both GPS and date validation
                logger.info(f"Final Results for pair {pair_index + 1}:")
                logger.info(f"  - Status: {status}")
                logger.info(f"  - Score: {score}")
                logger.info(f"  - GPS Valid: {gps_valid}")
                logger.info(f"  - Date Valid: {date_valid}")
                logger.info(f"  - Timeline Valid: {timeline_valid}")
                logger.info(f"  - Location Consistent: {location_consistent}")
                logger.info(f"  - Overall Valid: {metadata_valid}")
                logger.info(
                    f"  Status: {status} (score: {score}, metadata violations: {has_metadata_violations})"
                )

                # Convert images to base64 data URLs
                before_base64 = encode_image_to_base64(before_img["path"])
                after_base64 = encode_image_to_base64(after_img["path"])

                if not before_base64 or not after_base64:
                    logger.error(f"Failed to encode images for pair {pair_index + 1}")
                    continue

                # Updated result item to include both GPS and Date validations
                result_item = {
                    "pair_number": pair_index + 1,
                    "set_id": set_id,
                    "before": before_base64,
                    "after": after_base64,
                    "before_filename": before_img["original_filename"],
                    "after_filename": after_img["original_filename"],
                    "comparison_type": comparison_type,
                    "score": score,
                    "status": status,
                    "data": structured_data,
                    "html": result_html,
                    "validation_results": validation_results,  # GPS + Date validation
                    "metadata_valid": metadata_valid,
                    "gps_valid": gps_valid,
                    "date_valid": date_valid,
                    "timeline_valid": timeline_valid,
                    "location_consistent": location_consistent,
                    "metadata_violations": metadata_violations,
                    "metadata_validation_summary": {
                        "before_gps_valid": before_img.get(
                            "gps_coordinates_valid", False
                        ),
                        "after_gps_valid": after_img.get(
                            "gps_coordinates_valid", False
                        ),
                        "before_date_valid": before_img.get(
                            "date_compliance_valid", False
                        ),
                        "after_date_valid": after_img.get(
                            "date_compliance_valid", False
                        ),
                        "overall_valid": not has_metadata_violations,
                    },
                    # Add EXIF metadata for verification
                    "before_metadata": before_metadata,
                    "after_metadata": after_metadata,
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
                        "metadata_valid": metadata_valid,
                        "metadata_violations": metadata_violations,
                        "metadata_valid": not has_metadata_violations,
                    }
                )

                logger.info(f"✅ Successfully processed pair {pair_index + 1}")

            except Exception as e:
                logger.error(f"Error processing pair {pair_index + 1}: {e}")
                import traceback

                traceback.print_exc()
                continue

        if not results:
            return jsonify({"error": "Failed to process any image pairs"}), 500

        # Calculate statistics
        total_score = sum(r["score"] for r in results)
        average_score = total_score / len(results) if results else 0
        approved_count = sum(1 for r in results if r["status"] == "approved")
        metadata_valid_count = sum(1 for r in results if r["metadata_valid"])
        metadata_violations_count = sum(1 for r in results if r["metadata_violations"])

        # Count unique sets
        unique_sets = len(comparison_pairs)

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
                    metadata_valid_count=metadata_valid_count,
                    metadata_violations_count=metadata_violations_count,
                    timestamp=timestamp,
                    upload_count=len(uploaded_files),
                    unique_sets=unique_sets,
                    **BASE64_IMAGES,
                )
            )

        # Response data
        response_data = {
            "success": True,
            "message": f"Successfully processed {len(results)} comparison pair{'s' if len(results) > 1 else ''} with AI-powered GPS + Date validation",
            "message": f"Successfully processed {len(results)} comparison pair{'s' if len(results) > 1 else ''} from filename-based matching with metadata validation",
            "results": results,
            "total_pairs": len(results),
            "unique_sets": unique_sets,
            "report_type": "combined" if len(results) > 1 else "single",
            "filePath": report_filename,
            "average_score": round(average_score, 1),
            "approved_count": approved_count,
            "metadata_valid_count": metadata_valid_count,
            "validation_method": "ai_gps_date_validation",
            "metadata_violations_count": metadata_violations_count,
            "image_statistics": {
                "total_uploaded": len(uploaded_files),
                "total_processed": len(all_images_data),
                "unique_sets": unique_sets,
                "metadata_compliance_rate": (
                    f"{(metadata_valid_count / len(results) * 100):.1f}%"
                    if results
                    else "0%"
                ),
                    f"{((len(results) - metadata_violations_count) / len(results) * 100):.1f}%"
                    if results
                    else "0%"
                ),
            },
            "metadata_validation_summary": {
                "total_violations": metadata_violations_count,
                "gps_validation_enabled": True,
                "date_validation_enabled": True,
                "tolerance_meters": 100,  # Approximate GPS tolerance
            },
        }

        logger.info("=== FINAL RESPONSE SUMMARY ===")
        logger.info(f"Successfully processed: {len(results)} pairs")
        logger.info(f"Average score: {round(average_score, 1)}")
        logger.info(f"Approved: {approved_count}/{len(results)}")
        logger.info(f"Metadata valid: {metadata_valid_count}/{len(results)}")
        logger.info(f"Validation method: AI GPS + Date validation")

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Upload error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# =============================================================================
# PDF GENERATION AND UTILITY ROUTES
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "OutputFiles")


async def html_string_to_pdf_bytes(html_content):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(html_content, wait_until="networkidle")
        pdf_bytes = await page.pdf(
            format="A4",
            print_background=True,
            margin={
                "top": "0in",
                "right": "0in",
                "bottom": "0in",
                "left": "0in",
            },  # margins must be strings
        )
        await browser.close()
        return pdf_bytes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "OutputFiles")


@generatecomparison.route("/chatgenie/v1/pdf/<html_name>", methods=["GET"])
def generate_pdf(html_name):
    try:
        safe_filename = secure_filename(html_name)
        if not safe_filename.lower().endswith(".html"):
            safe_filename += ".html"

        file_path = os.path.join(OUTPUT_FOLDER, safe_filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "Report not found"}), 404

        # Read and parse HTML content
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # Inline CSS from local <link rel="stylesheet"> tags
        for link_tag in soup.find_all("link", rel="stylesheet"):
            css_href = link_tag.get("href", "")
            if css_href and not css_href.startswith(("http://", "https://")):
                css_path = os.path.join(OUTPUT_FOLDER, css_href)
                if os.path.exists(css_path):
                    with open(css_path, "r", encoding="utf-8") as css_file:
                        style_tag = soup.new_tag("style")
                        style_tag.string = css_file.read()
                        link_tag.replace_with(style_tag)

        # Fix image src attributes to absolute local file URLs for PDF rendering
        for img_tag in soup.find_all("img"):
            src = img_tag.get("src", "")
            if src and not src.startswith(("http://", "https://", "data:")):
                abs_img_path = os.path.abspath(os.path.join(OUTPUT_FOLDER, src))
                if os.path.exists(abs_img_path):
                    img_tag["src"] = "file:///" + abs_img_path.replace("\\", "/")

        final_html = str(soup)

        # Generate PDF bytes from HTML content asynchronously
        pdf_bytes = asyncio.run(html_string_to_pdf_bytes(final_html))

        pdf_filename = safe_filename.rsplit(".", 1)[0] + ".pdf"
        pdf_buffer = io.BytesIO(pdf_bytes)

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            download_name=pdf_filename,
            as_attachment=True,
        pdf_buffer.seek(0)
        pdf_filename = safe_filename.replace(".html", ".pdf")

        return send_file(
            pdf_buffer, mimetype="application/pdf", download_name=pdf_filename
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@generatecomparison.route("/chatgenie/v1/metadata/<filename>", methods=["GET"])
def get_metadata_endpoint(filename):
    """API endpoint to get metadata for a specific uploaded file"""
    try:
        safe_filename = secure_filename(filename)
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)

        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        # Extract metadata
        metadata = get_image_metadata(file_path)

        if not metadata:
            return jsonify({"error": "No metadata found for this image"}), 404

        # Get file info
        file_size = os.path.getsize(file_path)

        # Determine if this is an AFTER image
        is_after_image = safe_filename.lower().startswith("after_")

        response_data = {
            "filename": safe_filename,
            "file_size": file_size,
            "file_path": file_path,
            "is_after_image": is_after_image,
            "metadata": metadata,
            "extracted_data": {
                "gps_coordinates": {
                    "latitude": metadata.get("latitude"),
                    "longitude": metadata.get("longitude"),
                    "available": metadata.get("latitude") is not None
                    and metadata.get("longitude") is not None,
                },
                "datetime_info": {
                    "datetime_original": metadata.get("datetime_original"),
                    "datetime_created": metadata.get("datetime_created"),
                    "available": metadata.get("datetime_original") is not None,
                },
                "camera_info": metadata.get("camera_info", {}),
                "gps_raw": metadata.get("gps_info", {}),
            },
            "validation_ready": {
                "has_gps": metadata.get("latitude") is not None,
                "has_datetime": metadata.get("datetime_original") is not None,
                "ai_validation_enabled": True,  # AI handles all validation
            },
        }

        logger.info(f"Metadata extracted for {safe_filename}:")
        logger.info(f"  - GPS: {response_data['extracted_data']['gps_coordinates']}")
        logger.info(
            f"  - DateTime: {response_data['extracted_data']['datetime_info']['available']}"
        )
        logger.info(f"  - Is After Image: {is_after_image}")

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Metadata extraction API error: {e}")
        return jsonify({"error": str(e)}), 500


# =============================================================================
# ERROR HANDLERS
# =============================================================================
=======
        metadata = get_image_metadata(file_path)
        return jsonify(
            {
                "filename": safe_filename,
                "metadata": metadata,
                "file_size": os.path.getsize(file_path),
                "file_path": file_path,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@generatecomparison.route("/chatgenie/v1/validate-metadata", methods=["POST"])
def validate_metadata_endpoint():
    """API endpoint to validate metadata compliance"""
    try:
        data = request.get_json()

        form_lat = data.get("form_latitude")
        form_lon = data.get("form_longitude")
        exif_lat = data.get("exif_latitude")
        exif_lon = data.get("exif_longitude")
        image_datetime = data.get("image_datetime")
        issued_date = data.get("issued_date")
        completed_date = data.get("completed_date")

        gps_valid = validate_gps_coordinates(form_lat, form_lon, exif_lat, exif_lon)
        date_valid = validate_date_range(image_datetime, issued_date, completed_date)

        return jsonify(
            {
                "gps_coordinates_valid": gps_valid,
                "date_compliance_valid": date_valid,
                "overall_valid": gps_valid and date_valid,
                "validation_details": {
                    "gps_check": {
                        "form_coordinates": [form_lat, form_lon],
                        "exif_coordinates": [exif_lat, exif_lon],
                        "tolerance_degrees": 0.001,
                        "tolerance_meters_approx": 100,
                    },
                    "date_check": {
                        "image_datetime": image_datetime,
                        "work_period": [issued_date, completed_date],
                    },
                },
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@generatecomparison.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413


@generatecomparison.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500
