from pydantic import BaseModel, Field

class ModelForm(BaseModel):
    """Request form model for retrieving model parameters.
    
    This model defines the data structure required to request parameters of a specific model
    based on its ID and category.
    """
    model_id: int = Field(..., description="Model ID (unique numeric identifier of the target model)")
    model_category: int = Field(..., description="Model category (numeric code representing model type/classification)")