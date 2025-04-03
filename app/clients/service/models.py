from typing import List

from pydantic import BaseModel, Field


class PredictionFeatures(BaseModel):
    """Template class prediction class"""

    age: float = Field(..., description="Client's age", example=30)
    gender: float = Field(..., description="Client's gender (1 for male, 0 for female)", example=1)
    work_experience: float = Field(..., description="Years of work experience", example=5)
    canada_workex: float = Field(..., description="Years of work experience in Canada", example=2)
    dep_num: float = Field(..., description="Number of dependents", example=1)
    canada_born: float = Field(..., description="Born in Canada (1 for yes, 0 for no)", example=0)
    citizen_status: float = Field(..., description="Citizenship status", example=1)
    level_of_schooling: float = Field(..., description="Highest level achieved (1-14)", example=8)
    fluent_english: float = Field(..., description="English fluency scale (1-10)", example=7)
    reading_english_scale: float = Field(..., description="Reading ability scale (1-10)", example=6)
    speaking_english_scale: float = Field(
        ..., description="Speaking ability scale (1-10)", example=6
    )
    writing_english_scale: float = Field(..., description="Writing ability scale (1-10)", example=5)
    numeracy_scale: float = Field(..., description="Numeracy ability scale (1-10)", example=7)
    computer_scale: float = Field(..., description="Computer proficiency scale (1-10)", example=6)
    transportation_bool: float = Field(
        ..., description="Needs transportation support (1 for yes, 0 for no)", example=0
    )
    caregiver_bool: float = Field(
        ..., description="Is primary caregiver (1 for yes, 0 for no)", example=0
    )
    housing: float = Field(..., description="Housing situation (1-10)", example=3)
    income_source: float = Field(..., description="Source of income (1-10)", example=2)
    felony_bool: float = Field(..., description="Has a felony (1 for yes, 0 for no)", example=0)
    attending_school: float = Field(
        ..., description="Currently a student (1 for yes, 0 for no)", example=0
    )
    currently_employed: float = Field(
        ..., description="Currently employed (1 for yes, 0 for no)", example=0
    )
    substance_use: float = Field(
        ..., description="Substance use disorder (1 for yes, 0 for no)", example=0
    )
    time_unemployed: float = Field(..., description="Years unemployed", example=1)
    need_mental_health_support_bool: float = Field(
        ..., description="Needs mental health support (1 for yes, 0 for no)", example=0
    )
    # Intervention columns
    employment_assistance: float = Field(
        ..., description="Employment assistance intervention", example=1
    )
    life_stabilization: float = Field(..., description="Life stabilization intervention", example=1)
    retention_services: float = Field(..., description="Retention services intervention", example=0)
    specialized_services: float = Field(
        ..., description="Specialized services intervention", example=0
    )
    employment_related_financial_supports: float = Field(
        ..., description="Employment related financial supports", example=1
    )
    employer_financial_supports: float = Field(
        ..., description="Employer financial supports", example=0
    )
    enhanced_referrals: float = Field(..., description="Enhanced referrals", example=0)


class PredictionRequest(BaseModel):
    """Template class for prediction request"""

    features: List[float] = Field(
        ..., description="List of 31 features in specific order for model prediction"
    )

    @classmethod
    def from_structured_features(cls, structured_features: PredictionFeatures):
        features = [
            structured_features.age,
            structured_features.gender,
            structured_features.work_experience,
            structured_features.canada_workex,
            structured_features.dep_num,
            structured_features.canada_born,
            structured_features.citizen_status,
            structured_features.level_of_schooling,
            structured_features.fluent_english,
            structured_features.reading_english_scale,
            structured_features.speaking_english_scale,
            structured_features.writing_english_scale,
            structured_features.numeracy_scale,
            structured_features.computer_scale,
            structured_features.transportation_bool,
            structured_features.caregiver_bool,
            structured_features.housing,
            structured_features.income_source,
            structured_features.felony_bool,
            structured_features.attending_school,
            structured_features.currently_employed,
            structured_features.substance_use,
            structured_features.time_unemployed,
            structured_features.need_mental_health_support_bool,
            structured_features.employment_assistance,
            structured_features.life_stabilization,
            structured_features.retention_services,
            structured_features.specialized_services,
            structured_features.employment_related_financial_supports,
            structured_features.employer_financial_supports,
            structured_features.enhanced_referrals,
        ]
        return cls(features=features)
