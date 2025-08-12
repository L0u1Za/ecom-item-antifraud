def check_item_name_consistency(item_name, description):
    """
    Check if the item name is consistent with the description.
    Returns True if consistent, False otherwise.
    """
    return item_name.lower() in description.lower()

def check_description_length(description, min_length=10):
    """
    Check if the description meets the minimum length requirement.
    Returns True if valid, False otherwise.
    """
    return len(description) >= min_length

def validate_item(item_name, description):
    """
    Validate the item based on business rules.
    Returns a dictionary with validation results.
    """
    validation_results = {
        "name_consistency": check_item_name_consistency(item_name, description),
        "description_length": check_description_length(description)
    }
    return validation_results

def detect_inconsistencies(item_name, description):
    """
    Detect inconsistencies in item name and description.
    Returns a list of issues found.
    """
    issues = []
    if not check_item_name_consistency(item_name, description):
        issues.append("Item name does not match description.")
    if not check_description_length(description):
        issues.append("Description is too short.")
    
    return issues