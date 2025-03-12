import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Gemma2ForTokenClassification, BitsAndBytesConfig

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")

def repeat_function(xs, max_length = 128):
    new_xs = []
    for x in xs:
        if x.shape[1] >= max_length-1:
            new_xs.append(x[:,:max_length-1,:])
        else:
            new_xs.append(x)
    xs = new_xs
    mean_xs = [x.mean(1,keepdim=True).expand(-1,max_length - x.shape[1],-1) for x in xs]
    xs = [torch.cat([x,mean_x],1) for mean_x, x in zip(mean_xs, xs)]
    return xs

class Gemma2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", )
        self.tokenizer_max_length = 128
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = Gemma2ForTokenClassification.from_pretrained(
            "google/gemma-2-2b",
            # device_map="auto",
            # quantization_config=quantization_config,
        ).float()
        self.model.score = nn.Identity()
    
    @torch.no_grad()
    def forward(self, input_prompt):
        input_prompt = list(input_prompt)
        outputs = []
        for _input_prompt in input_prompt:
            input_ids = self.tokenizer(_input_prompt, add_special_tokens=False, max_length=77, return_tensors="pt").to("cuda")
            _outputs = self.model(**input_ids)["logits"]
            outputs.append(_outputs)
        outputs = repeat_function(outputs)
        outputs = torch.cat(outputs,0)
        return outputs

if __name__ == "__main__":
    model = Gemma2Model().cuda()
    input_text = ["Write me a poem about Machine Learning.", "Write me a poem about Deep Learning."]
    print(model(input_text))
    print(model(input_text)[0].shape)
    print(model(input_text).shape)