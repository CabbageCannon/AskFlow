"""Model initialization helpers."""

from typing import Any
from open_deep_research.model_router import ModelDecision

def get_provider_model_kwargs(
  model_name:str,
  *,
  thinking_enabled:bool=False
)->dict[str,Any]:
  """Return provider-specific model configuration."""
  
  normalized_name=model_name.lower()
  
  if normalized_name.startswith("deepseek:"):
    return{
      "extra_body":{
        "thinking":{
          "type":"enabled" if thinking_enabled else "disabled"
        }
      }
    }
  
  return {}


# "model": configurable.research_model,
# "max_tokens": configurable.research_model_max_tokens,
# "api_key": get_api_key_for_model(configurable.research_model, config),
# **get_provider_model_config(
#     configurable.research_model
# ),
# "tags": ["langsmith:nostream"],
def build_model_runtime_config(
  *,
  model_name:str,
  max_tokens:int,
  api_key:str|None,
  base_url:str|None=None,
  extra_body:dict[str,Any]|None=None,
  no_stream:bool=True
)->dict[str,Any]:
  """Build runtime configuration for a configurable chat model."""
  
  runtime_config:dict[str,Any]={
    "model":model_name,
    "max_tokens":max_tokens,
    **get_provider_model_kwargs(model_name)
  }
  
  if api_key is not None:
    runtime_config["api_key"]=api_key
    
  if base_url is not None:
        runtime_config["base_url"] = base_url
    
  if extra_body is not None:
        runtime_config["extra_body"] = extra_body
        
  if no_stream:
    runtime_config["tags"]=["langsmith:nostream"]
    
  return runtime_config
  
def build_routed_model_runtime_config(
    decision: ModelDecision,
    *,
    api_key: str | None,
    base_url: str,
    no_stream: bool = True,
) -> dict[str, Any]:
    """Convert a router decision into LangChain runtime configuration."""

    if not api_key:
        raise ValueError(
            "BAILIAN_API_KEY is required for routed model calls."
        )

    model = decision.model

    return build_model_runtime_config(
        model_name=model.model_name,
        max_tokens=model.max_tokens,
        api_key=api_key,
        base_url=base_url,
        extra_body=model.extra_body,
        no_stream=no_stream,
    )