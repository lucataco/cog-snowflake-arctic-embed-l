# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input
from typing import List
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Snowflake/snowflake-arctic-embed-l"
MODEL_CACHE = "checkpoint"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE)
        self.model = AutoModel.from_pretrained(MODEL_ID, add_pooling_layer=False, cache_dir=MODEL_CACHE)

    def predict(
        self,
        prompt: str = Input(description="Prompt to generate a vector embedding for", default='Snowflake is the Data Cloud!'),
    ) -> List[float]:
        """Run a single prediction on the model"""
        documents = [prompt]
        encoded_input = self.tokenizer(documents, padding=True, return_tensors='pt')
        outputs = self.model(**encoded_input).last_hidden_state
        embeddings = outputs[:, 0].tolist()[0]
        return embeddings
