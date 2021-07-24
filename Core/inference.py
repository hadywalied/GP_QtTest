import torch
from PySide2.QtCore import Signal, QObject
from transformers import AutoTokenizer, PegasusForConditionalGeneration, AutoModelForSeq2SeqLM
from torch.quantization import quantize_dynamic

modelspaths = {'Pegasus-xsum  16-16': ('E:\\GP Models\\teacher_16_16', 24.6),
               # 'Pegasus-xsum  TA/KD 16-12': ('E:\\GP Models\\pipeline\\xsum 16-12\\best_tfmr', 22.1),
               # 'Pegasus-xsum  TA/KD 16-8': ('E:\\GP Models\\pipeline\\xsum 16-8_\\best_tfmr', 21.3),
               'Pegasus-xsum  TA/KD 16-4': ('E:\\GP Models\\pipeline\\xsum 16-4\\best_tfmr', 15.75),
               # 'Pegasus-xsum  SF 16-4': (''),
               'Pegasus-xsum  PL 16-4': ('E:\\GP Models\\teacher_16_4.pt', 21.92),
               'Pegasus-xsum  PL 16-2': ('E:\\GP Models\\trained_student16_2_1.pt', 21.2),
               'Pegasus-xsum  PL 12-3': ('E:\\GP Models\\trained_student_12ecn_3dec_1.pt', 17.2),
               'Pegasus-xsum  PL 12-3 unfreezed': (
                   'E:\\GP Models\\trained_student_12ecn_3dec_unfreeze_last2_1.pt', 18.65)}
from contextlib import contextmanager
import time
import warnings

warnings.filterwarnings('ignore')


@contextmanager
def timer(msg):
    t0 = time.time()
    print(f'[{msg}] start.')
    yield
    elapsed_time = time.time() - t0
    print(f'[{msg}] done in {elapsed_time} sec.')


class InferenceClass(QObject):

    def __init__(self):
        QObject.__init__(self)

    def infer(self, text, models, quantized):
        model_ckpt = modelspaths[models.text()][0]
        Rouge_Score = modelspaths[models.text()][1]
        max_input_length = 512

        model = self.getModel(model_ckpt, max_input_length, quantized)
        model.eval()
        tokens = self.tokenize(text=text, tokenizerName='google/pegasus-xsum',
                               max_input_length=max_input_length)

        output_tokens, elapsed_time = self.generateSummary(model=model, tokens=tokens)

        output = self.decodeOutput(tokenizerName='google/pegasus-xsum', tokens=output_tokens)

        del model
        import gc
        gc.collect()
        return output, elapsed_time, Rouge_Score

    def getModel(self, path, max_input_length, quantized):
        with timer('Loading Model'):
            if (path.__contains__(".pt")):
                model = torch.load(path, map_location=torch.device('cpu'))
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(path)

        if quantized:
            with timer('Quantize the Model'):
                model = quantize_dynamic(model)
                # model.save_pretrained('E:\\GP Models\\quantized_16_8')

        return model

    def tokenize(self, text, tokenizerName, max_input_length):
        with timer('Tokenizing ...'):
            tokenizer = AutoTokenizer.from_pretrained(tokenizerName)
            token = tokenizer(text, truncation=True, padding='max_length', max_length=max_input_length,
                              return_tensors="pt")
        return token

    def generateSummary(self, model, tokens):
        with timer('Generating Summary ...'):
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            # 'set num_beams = 1' for greedy search
            t0 = time.time()
            tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=4)
            elapsed_time = time.time() - t0

        return tokens, elapsed_time

    def decodeOutput(self, tokenizerName, tokens):
        with timer('Decoding Output'):
            tokenizer = AutoTokenizer.from_pretrained(tokenizerName)
            output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
        return output
