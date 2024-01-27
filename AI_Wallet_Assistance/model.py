from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

my_credentials = {
    "url"    : "https://us-south.ml.cloud.ibm.com",
    "apikey" : "zayEviSjzRrJ1KByBL3ktMbigosqRIteouf-suHh1Okw" # <--- NOTE: replace "YOUR_WATSONX_API" with your actual API key)
}

params = {
        GenParams.MAX_NEW_TOKENS: 256, # The maximum number of tokens that the model can generate in a single run.
        GenParams.TEMPERATURE: 0.1,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
    }

LLAMA2_model = Model(
        model_id= 'meta-llama/llama-2-70b-chat',
        credentials=my_credentials,
        params=params,
        project_id="bcef5b8a-8c80-47bd-88ea-3ed83be66bf4",  # <--- NOTE: replace "YOUR_PROJECT_ID" with your actual project ID
        )

llm = WatsonxLLM(LLAMA2_model)
