"""
GST Rules Engine — GSTIN validation, HSN database, and tax computation.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# GSTIN Validation
# ---------------------------------------------------------------------------

GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

VALID_STATE_CODES = {
    "01": "Jammu & Kashmir",
    "02": "Himachal Pradesh",
    "03": "Punjab",
    "04": "Chandigarh",
    "05": "Uttarakhand",
    "06": "Haryana",
    "07": "Delhi",
    "08": "Rajasthan",
    "09": "Uttar Pradesh",
    "10": "Bihar",
    "11": "Sikkim",
    "12": "Arunachal Pradesh",
    "13": "Nagaland",
    "14": "Manipur",
    "15": "Mizoram",
    "16": "Tripura",
    "17": "Meghalaya",
    "18": "Assam",
    "19": "West Bengal",
    "20": "Jharkhand",
    "21": "Odisha",
    "22": "Chhattisgarh",
    "23": "Madhya Pradesh",
    "24": "Gujarat",
    "25": "Daman & Diu",
    "26": "Dadra & Nagar Haveli and Daman & Diu",
    "27": "Maharashtra",
    "28": "Andhra Pradesh (old)",
    "29": "Karnataka",
    "30": "Goa",
    "31": "Lakshadweep",
    "32": "Kerala",
    "33": "Tamil Nadu",
    "34": "Puducherry",
    "35": "Andaman & Nicobar Islands",
    "36": "Telangana",
    "37": "Andhra Pradesh",
}


def _luhn_mod36_checksum(input_str: str) -> str:
    """Compute the Luhn mod-36 checksum character for a 14-character GSTIN prefix."""
    factor = 2
    total = 0
    for ch in reversed(input_str):
        value = GSTIN_CHARS.index(ch) * factor
        value = (value // 36) + (value % 36)
        total += value
        factor = 3 if factor == 2 else 2
    remainder = total % 36
    check_digit = (36 - remainder) % 36
    return GSTIN_CHARS[check_digit]


def validate_gstin(gstin: str) -> dict:
    """
    Validate a GSTIN string.

    Returns {"valid": bool, "reason": str}.
    """
    if not isinstance(gstin, str):
        return {"valid": False, "reason": "GSTIN must be a string"}

    gstin = gstin.strip().upper()

    if len(gstin) != 15:
        return {"valid": False, "reason": f"GSTIN must be 15 characters, got {len(gstin)}"}

    state_code = gstin[:2]
    if state_code not in VALID_STATE_CODES:
        return {"valid": False, "reason": f"Invalid state code: {state_code}"}

    # Positions 3-12 (0-indexed 2-11): 10-char PAN → 5 letters + 4 digits + 1 letter
    pan = gstin[2:12]
    pan_pattern = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
    if not pan_pattern.match(pan):
        return {"valid": False, "reason": f"Invalid PAN format in GSTIN: {pan}"}

    # Position 13 (0-indexed 12): entity code, alphanumeric 1-9 or A-Z
    entity_code = gstin[12]
    if entity_code not in "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return {"valid": False, "reason": f"Invalid entity code: {entity_code}"}

    # Position 14 (0-indexed 13): must be 'Z'
    if gstin[13] != "Z":
        return {"valid": False, "reason": f"Position 14 must be 'Z', got '{gstin[13]}'"}

    # Position 15 (0-indexed 14): checksum
    expected_checksum = _luhn_mod36_checksum(gstin[:14])
    if gstin[14] != expected_checksum:
        return {
            "valid": False,
            "reason": f"Checksum mismatch: expected '{expected_checksum}', got '{gstin[14]}'",
        }

    return {"valid": True, "reason": "Valid GSTIN"}


def generate_valid_gstin(state_code: str, pan: str, entity: str = "1") -> str:
    """
    Generate a valid GSTIN with the correct Luhn mod-36 checksum.

    Args:
        state_code: Two-digit state code, e.g. "27"
        pan: 10-character PAN string
        entity: Entity code character (default "1")

    Returns:
        A 15-character valid GSTIN string.
    """
    if len(state_code) != 2 or state_code not in VALID_STATE_CODES:
        raise ValueError(f"Invalid state code: {state_code}")
    pan = pan.upper().strip()
    pan_pattern = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
    if not pan_pattern.match(pan):
        raise ValueError(f"Invalid PAN format: {pan}")
    entity = entity.upper().strip()
    if entity not in "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        raise ValueError(f"Invalid entity code: {entity}")

    prefix = f"{state_code}{pan}{entity}Z"  # 14 characters
    checksum = _luhn_mod36_checksum(prefix)
    return prefix + checksum


# ---------------------------------------------------------------------------
# HSN Database
# ---------------------------------------------------------------------------

HSN_DATABASE: dict = {
    # ---- Food & Beverages (0%) ----
    "01011000": {
        "description": "Live horses",
        "tax_rate": 0,
        "keywords": ["horse", "live horse", "equine"],
    },
    "01012100": {
        "description": "Pure-bred breeding horses",
        "tax_rate": 0,
        "keywords": ["breeding horse", "pure-bred horse"],
    },
    "02011000": {
        "description": "Fresh bovine carcasses and half-carcasses",
        "tax_rate": 0,
        "keywords": ["beef", "bovine", "carcass", "meat bovine fresh"],
    },
    "04011000": {
        "description": "Fresh milk (fat content ≤ 1%)",
        "tax_rate": 0,
        "keywords": ["milk", "fresh milk", "dairy milk", "toned milk"],
    },
    "07019000": {
        "description": "Fresh potatoes (other)",
        "tax_rate": 0,
        "keywords": ["potato", "fresh potato", "aloo"],
    },
    "07031000": {
        "description": "Fresh onions and shallots",
        "tax_rate": 0,
        "keywords": ["onion", "pyaz", "shallot", "fresh onion"],
    },
    "07070000": {
        "description": "Fresh cucumbers and gherkins",
        "tax_rate": 0,
        "keywords": ["cucumber", "kheera", "gherkin"],
    },
    "10011100": {
        "description": "Durum wheat seed",
        "tax_rate": 0,
        "keywords": ["wheat seed", "durum wheat", "gehu"],
    },
    "10061010": {
        "description": "Rice in husk (paddy)",
        "tax_rate": 0,
        "keywords": ["paddy", "rice husk", "raw rice", "dhan"],
    },
    "11010000": {
        "description": "Wheat or meslin flour",
        "tax_rate": 0,
        "keywords": ["atta", "wheat flour", "flour", "ite flour"],
    },
    # ---- Food & Beverages (5%) ----
    "09011100": {
        "description": "Coffee, not roasted, not decaffeinated",
        "tax_rate": 5,
        "keywords": ["coffee", "coffee beans", "raw coffee"],
    },
    "09021000": {
        "description": "Green tea (not fermented)",
        "tax_rate": 5,
        "keywords": ["green tea", "tea leaves", "chai"],
    },
    "10063000": {
        "description": "Semi-milled or wholly milled rice",
        "tax_rate": 5,
        "keywords": ["rice", "milled rice", "polished rice", "chawal"],
    },
    "17011200": {
        "description": "Cane sugar",
        "tax_rate": 5,
        "keywords": ["sugar", "cane sugar", "cheeni", "shakkar"],
    },
    "19049000": {
        "description": "Cereal preparations (other)",
        "tax_rate": 5,
        "keywords": ["cereal", "corn flakes", "breakfast cereal", "muesli"],
    },
    "20081100": {
        "description": "Groundnuts, prepared or preserved",
        "tax_rate": 5,
        "keywords": ["peanut", "groundnut", "mungfali", "roasted peanut"],
    },
    # ---- Food & Beverages (12%) ----
    "17049000": {
        "description": "Sugar confectionery (other)",
        "tax_rate": 12,
        "keywords": ["candy", "sweets", "confectionery", "mithai", "toffee"],
    },
    "18063100": {
        "description": "Chocolate filled with fillings",
        "tax_rate": 12,
        "keywords": ["chocolate", "filled chocolate", "candy bar"],
    },
    "20093100": {
        "description": "Juice of any single citrus fruit",
        "tax_rate": 12,
        "keywords": ["juice", "orange juice", "fruit juice", "citrus juice"],
    },
    "21039000": {
        "description": "Sauces and condiments",
        "tax_rate": 12,
        "keywords": ["sauce", "ketchup", "chutney", "condiment", "relish"],
    },
    # ---- Food & Beverages (18%) ----
    "21069099": {
        "description": "Food preparations not elsewhere classified",
        "tax_rate": 18,
        "keywords": ["food supplement", "protein powder", "health supplement"],
    },
    "22011000": {
        "description": "Mineral waters and aerated waters",
        "tax_rate": 18,
        "keywords": ["mineral water", "aerated water", "sparkling water", "soda"],
    },
    "22021000": {
        "description": "Waters with added sugar or flavouring",
        "tax_rate": 18,
        "keywords": ["soft drink", "flavoured water", "sweetened water"],
    },
    # ---- Food & Beverages (28%) ----
    "22030000": {
        "description": "Beer made from malt",
        "tax_rate": 28,
        "keywords": ["beer", "malt beer", "lager", "ale"],
    },
    "22041000": {
        "description": "Sparkling wine",
        "tax_rate": 28,
        "keywords": ["sparkling wine", "champagne", "prosecco"],
    },
    "24011000": {
        "description": "Tobacco, not stemmed/stripped",
        "tax_rate": 28,
        "keywords": ["tobacco", "raw tobacco", "cigarette tobacco"],
    },
    # ---- Textiles (5%) ----
    "52081100": {
        "description": "Plain woven fabrics of cotton, ≥ 85% cotton, ≤ 100 g/m²",
        "tax_rate": 5,
        "keywords": ["cotton fabric", "woven cotton", "cotton cloth"],
    },
    "54071000": {
        "description": "Woven fabrics of synthetic filament yarn",
        "tax_rate": 5,
        "keywords": ["synthetic fabric", "polyester fabric", "nylon fabric"],
    },
    "61091000": {
        "description": "T-shirts, singlets, cotton, knitted",
        "tax_rate": 5,
        "keywords": ["t-shirt", "tshirt", "cotton shirt", "vest"],
    },
    # ---- Textiles (12%) ----
    "62034200": {
        "description": "Men's trousers of cotton",
        "tax_rate": 12,
        "keywords": ["trousers", "pants", "cotton trousers", "men trousers"],
    },
    "62044200": {
        "description": "Women's suits and dresses of cotton",
        "tax_rate": 12,
        "keywords": ["women dress", "ladies dress", "cotton dress", "salwar"],
    },
    # ---- Chemicals & Pharma (12%) ----
    "29054500": {
        "description": "Glycerol (pharma grade)",
        "tax_rate": 12,
        "keywords": ["glycerol", "glycerin", "glycerine"],
    },
    "30039000": {
        "description": "Medicaments (other, not put in measured doses)",
        "tax_rate": 12,
        "keywords": ["medicine", "medicament", "drug", "pharmaceutical"],
    },
    "30049000": {
        "description": "Medicaments in measured doses (other)",
        "tax_rate": 12,
        "keywords": ["tablet", "capsule", "syrup", "injection", "medicine dose"],
    },
    # ---- Chemicals & Pharma (18%) ----
    "28042100": {
        "description": "Argon",
        "tax_rate": 18,
        "keywords": ["argon", "noble gas", "inert gas"],
    },
    "29011000": {
        "description": "Acyclic hydrocarbons, saturated",
        "tax_rate": 18,
        "keywords": ["methane", "ethane", "propane", "hydrocarbon"],
    },
    "33051000": {
        "description": "Shampoos",
        "tax_rate": 18,
        "keywords": ["shampoo", "hair wash", "hair cleanser"],
    },
    "33061000": {
        "description": "Dentifrices (toothpaste)",
        "tax_rate": 18,
        "keywords": ["toothpaste", "dentifrice", "tooth powder"],
    },
    # ---- Plastics & Rubber (18%) ----
    "39011000": {
        "description": "Polyethylene with specific gravity < 0.94",
        "tax_rate": 18,
        "keywords": ["polyethylene", "LDPE", "polythene", "plastic film"],
    },
    "39231000": {
        "description": "Boxes, cases, crates of plastics",
        "tax_rate": 18,
        "keywords": ["plastic box", "plastic crate", "plastic case", "container plastic"],
    },
    # ---- Plastics & Rubber (28%) ----
    "40111000": {
        "description": "New pneumatic tyres of rubber, motor cars",
        "tax_rate": 28,
        "keywords": ["tyre", "tire", "car tyre", "rubber tyre", "pneumatic tyre"],
    },
    "40151100": {
        "description": "Surgical gloves of vulcanised rubber",
        "tax_rate": 28,
        "keywords": ["rubber gloves", "surgical gloves", "latex gloves"],
    },
    # ---- Iron & Steel (18%) ----
    "72041000": {
        "description": "Waste and scrap of cast iron",
        "tax_rate": 18,
        "keywords": ["cast iron scrap", "iron scrap", "cast iron waste"],
    },
    "72081000": {
        "description": "Flat-rolled products of iron/non-alloy steel, hot-rolled",
        "tax_rate": 18,
        "keywords": ["steel sheet", "iron sheet", "HR coil", "hot rolled steel"],
    },
    "72142000": {
        "description": "Bars and rods of iron or non-alloy steel, with indentations",
        "tax_rate": 18,
        "keywords": ["TMT bar", "steel bar", "rebar", "iron rod", "sariya"],
    },
    # ---- Machinery (18%) ----
    "84151000": {
        "description": "Air conditioning machines for windows or walls",
        "tax_rate": 18,
        "keywords": ["air conditioner", "window ac unit", "split ac unit", "air conditioning unit"],
    },
    "84181000": {
        "description": "Combined refrigerator-freezers",
        "tax_rate": 18,
        "keywords": ["refrigerator", "fridge", "freezer", "double door fridge"],
    },
    "84501100": {
        "description": "Washing machines, fully automatic",
        "tax_rate": 18,
        "keywords": ["washing machine", "laundry machine", "washer"],
    },
    "84713000": {
        "description": "Portable data processing machines (laptops)",
        "tax_rate": 18,
        "keywords": ["laptop", "notebook computer", "portable computer"],
    },
    # ---- Electronics (12%) ----
    "85176200": {
        "description": "Machines for reception and transmission of voice/data",
        "tax_rate": 12,
        "keywords": ["router", "modem", "network equipment", "wifi router"],
    },
    "85258000": {
        "description": "Television cameras, digital cameras, video recorders",
        "tax_rate": 12,
        "keywords": ["camera", "digital camera", "video camera", "DSLR", "webcam"],
    },
    # ---- Electronics (18%) ----
    "85171100": {
        "description": "Line telephone sets with cordless handsets",
        "tax_rate": 18,
        "keywords": ["telephone", "landline phone", "cordless phone"],
    },
    "85171200": {
        "description": "Telephones for cellular networks (mobile phones)",
        "tax_rate": 18,
        "keywords": ["mobile phone", "smartphone", "cell phone", "handset"],
    },
    "85219000": {
        "description": "Video recording apparatus",
        "tax_rate": 18,
        "keywords": ["DVD player", "Blu-ray player", "video recorder", "set-top box"],
    },
    # ---- Electronics (28%) ----
    "85284900": {
        "description": "Monitors and projectors, not incorporating TV reception apparatus",
        "tax_rate": 28,
        "keywords": ["monitor", "computer monitor", "LED monitor", "projector display"],
    },
    "85291000": {
        "description": "Aerials and antennae",
        "tax_rate": 28,
        "keywords": ["antenna", "aerial", "dish antenna", "TV antenna"],
    },
    # ---- Vehicles (28%) ----
    "87032200": {
        "description": "Motor cars with engine capacity 1000–1500 cc",
        "tax_rate": 28,
        "keywords": ["car", "motor car", "passenger car", "sedan", "hatchback"],
    },
    "87112010": {
        "description": "Motorcycles with engine capacity 50–250 cc",
        "tax_rate": 28,
        "keywords": ["motorcycle", "bike", "two-wheeler", "motorbike", "scooter"],
    },
    # ---- Furniture (18%) ----
    "94011000": {
        "description": "Seats of a kind used for aircraft",
        "tax_rate": 18,
        "keywords": ["aircraft seat", "plane seat", "aviation seat"],
    },
    "94036000": {
        "description": "Wooden furniture (other)",
        "tax_rate": 18,
        "keywords": ["wooden furniture", "wood furniture", "timber furniture", "wooden table", "wooden chair"],
    },
    # ---- Furniture (28%) ----
    "94034000": {
        "description": "Wooden furniture of a kind used in kitchens",
        "tax_rate": 28,
        "keywords": ["kitchen cabinet", "kitchen furniture", "modular kitchen", "wood kitchen"],
    },
    "94051000": {
        "description": "Chandeliers and ceiling/wall lighting fittings (electric)",
        "tax_rate": 28,
        "keywords": ["chandelier", "ceiling light", "wall light", "light fitting", "lamp"],
    },
    # ---- Services / SAC codes (5%) ----
    "99631000": {
        "description": "Services by way of transportation of goods by road",
        "tax_rate": 5,
        "keywords": ["road transport", "truck transport", "goods transport", "freight road"],
    },
    "99635000": {
        "description": "Services by way of transportation of goods by air",
        "tax_rate": 5,
        "keywords": ["air freight", "cargo air", "air transport goods", "air cargo"],
    },
    # ---- Services / SAC codes (18%) ----
    "99831110": {
        "description": "Legal advisory and representation services",
        "tax_rate": 18,
        "keywords": ["legal service", "lawyer", "advocate", "legal advice", "legal consulting"],
    },
    "99831200": {
        "description": "Accounting and bookkeeping services",
        "tax_rate": 18,
        "keywords": ["accounting", "bookkeeping", "auditing", "CA services", "financial accounting"],
    },
    "99832000": {
        "description": "Management consulting services",
        "tax_rate": 18,
        "keywords": ["consulting", "management consulting", "business consulting", "strategy consulting"],
    },
    "99833000": {
        "description": "IT design and development services",
        "tax_rate": 18,
        "keywords": ["software development", "IT services", "web development", "app development", "programming"],
    },
}


# ---------------------------------------------------------------------------
# HSN Lookup
# ---------------------------------------------------------------------------

def lookup_hsn(product_description: str) -> Optional[dict]:
    """
    Look up an HSN entry by keyword matching against product_description.

    Returns {"hsn_code", "description", "tax_rate"} or None if no match.
    """
    if not product_description:
        return None

    description_lower = product_description.lower()

    best_match_code = None
    best_match_count = 0

    for hsn_code, entry in HSN_DATABASE.items():
        match_count = 0
        for keyword in entry["keywords"]:
            if keyword.lower() in description_lower:
                match_count += 1
        if match_count > best_match_count:
            best_match_count = match_count
            best_match_code = hsn_code

    if best_match_code is None or best_match_count == 0:
        return None

    entry = HSN_DATABASE[best_match_code]
    return {
        "hsn_code": best_match_code,
        "description": entry["description"],
        "tax_rate": entry["tax_rate"],
    }


# ---------------------------------------------------------------------------
# Tax Computation
# ---------------------------------------------------------------------------

def compute_tax(
    taxable_value: float,
    tax_rate: int,
    business_state: str,
    place_of_supply: str,
) -> dict:
    """
    Compute GST on a taxable value.

    Args:
        taxable_value: The base value (before tax).
        tax_rate: GST rate as integer percentage (0, 5, 12, 18, or 28).
        business_state: State code or name of the seller.
        place_of_supply: State code or name of the buyer/delivery.

    Returns:
        {
            "cgst": float,       # intra-state only
            "sgst": float,       # intra-state only
            "igst": float,       # inter-state only
            "cess": float,       # always 0.0 (not implemented)
            "total_tax": float,
            "is_interstate": bool,
        }
    """
    is_interstate = business_state.strip().lower() != place_of_supply.strip().lower()

    total_tax = round(taxable_value * tax_rate / 100, 2)

    if is_interstate:
        igst = total_tax
        cgst = 0.0
        sgst = 0.0
    else:
        igst = 0.0
        # Each component is exactly half, both rounded to 2 dp
        cgst = round(total_tax / 2, 2)
        sgst = round(total_tax / 2, 2)
        # Correct for any rounding residual (e.g. Rs 0.01 discrepancy)
        residual = round(total_tax - cgst - sgst, 2)
        if residual != 0.0:
            cgst = round(cgst + residual, 2)

    return {
        "cgst": cgst,
        "sgst": sgst,
        "igst": igst,
        "cess": 0.0,
        "total_tax": total_tax,
        "is_interstate": is_interstate,
    }
