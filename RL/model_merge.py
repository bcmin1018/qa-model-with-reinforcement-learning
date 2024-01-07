from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

access_token=""
base_model_name = ""
adapter_model_name = ""
output_name = ""

def model_merge(base_model_name, adapter_model_name, output_name):

  model = AutoModelForCausalLM.from_pretrained(
      base_model_name
  )

  # Load the Lora model
  model = PeftModel.from_pretrained(model, adapter_model_name)
  model.eval()

  model = model.merge_and_unload()
  model.save_pretrained(output_name)
  model.push_to_hub(output_name, use_temp_dir=True, use_auth_token=access_token)

model_merge(base_model_name, adapter_model_name, output_name)