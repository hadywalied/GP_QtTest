from PySide2.QtCore import Signal, QObject
from transformers import AutoTokenizer, PegasusForConditionalGeneration
from torch.quantization import quantize_dynamic

import asyncio

modelspaths = {'Pegasus-xsum  16-16': 'D:\\Pegasus_Model\\saves\\pegasus-gigaword',
               'Pegasus-xsum  SF 16-12': '',
               'Pegasus-xsum  SF 16-8': '',
               'Pegasus-xsum  SF 16-4': '',
               'Pegasus-xsum  PL 16-4': '',
               'Pegasus-xsum  PL 12-6': '',
               'Pegasus-xsum  PL 12-3': ''}


class InferenceClass(QObject):
    outputSignal = Signal(str)
    toggleProgressAndButton = Signal(bool)
    loggingSignal = Signal(str)

    def __init__(self):
        QObject.__init__(self)

    async def infer(self, text, model, quantized):
        model_ckpt = modelspaths[model.text()]
        max_input_length = 128
        model = await self.getModel(model_ckpt, max_input_length, quantized)
        tokens = await self.tokenize(text=text, tokenizerName='google/pegasus-gigaword',
                                     max_input_length=max_input_length)
        output_tokens = await self.generateSummary(model=model, tokens=tokens)
        output = await self.decodeOutput(tokenizerName='google/pegasus-gigaword', tokens=output_tokens)
        self.emitOutput(output)

    async def getModel(self, path, max_input_length, quantized):
        model = PegasusForConditionalGeneration.from_pretrained(path, max_length=max_input_length,
                                                                max_position_embeddings=max_input_length)
        if quantized:
            model = quantize_dynamic(model)
        return model

    async def tokenize(self, text, tokenizerName, max_input_length):
        tokenizer = AutoTokenizer.from_pretrained(tokenizerName)
        token = tokenizer(text, truncation=True, padding='max_length', max_length=max_input_length,
                          return_tensors="pt")
        return token

    async def generateSummary(self, model, tokens):
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        # 'set num_beams = 1' for greedy search
        tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=4)
        return tokens

    async def decodeOutput(self, tokenizerName, tokens):
        tokenizer = AutoTokenizer.from_pretrained(tokenizerName)
        output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
        return output

    def emitOutput(self, outputSummary):
        self.outputSignal.emit(outputSummary)
