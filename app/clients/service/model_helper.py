def get_feature_columns():
    """Get all feature columns"""
    return [
        "age",
        "gender",
        "work_experience",
        "canada_workex",
        "dep_num",
        "canada_born",
        "citizen_status",
        "level_of_schooling",
        "fluent_english",
        "reading_english_scale",
        "speaking_english_scale",
        "writing_english_scale",
        "numeracy_scale",
        "computer_scale",
        "transportation_bool",
        "caregiver_bool",
        "housing",
        "income_source",
        "felony_bool",
        "attending_school",
        "currently_employed",
        "substance_use",
        "time_unemployed",
        "need_mental_health_support_bool",
    ]


def get_intervention_columns():
    """Get all intervention columns"""
    return [
        "employment_assistance",
        "life_stabilization",
        "retention_services",
        "specialized_services",
        "employment_related_financial_supports",
        "employer_financial_supports",
        "enhanced_referrals",
    ]


def get_all_feature_columns():
    """Get all feature columns"""
    return get_feature_columns() + get_intervention_columns()


def get_true_file_name(model_type, filename):
    """Format pikle file name"""
    return filename.format(model_type).replace(" ", "_")
