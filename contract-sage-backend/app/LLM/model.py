# # # # === model.py ===
# # # import torch.nn as nn

# # # class LegalSemanticSegmentation(nn.Module):
# # #     def __init__(self, bert_model, num_labels=7):
# # #         super(LegalSemanticSegmentation, self).__init__()
# # #         self.bert = bert_model
# # #         self.dropout = nn.Dropout(0.1)
# # #         self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)

# # #     def forward(self, input_ids, attention_mask=None, token_type_ids=None):
# # #         outputs = self.bert(
# # #             input_ids=input_ids,
# # #             attention_mask=attention_mask,
# # #             token_type_ids=token_type_ids,
# # #             return_dict=True
# # #         )
# # #         cls_output = outputs.last_hidden_state[:, 0, :]
# # #         cls_output = self.dropout(cls_output)
# # #         logits = self.classifier(cls_output)
# # #         return logits


# # # model.py
# # # import torch
# # # import torch.nn as nn
# # # from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification
# # # from peft import PeftModel
# # # from typing import Dict, Any, List, Tuple, Optional
# # # import logging
# # # from .config import config


# # # # Set up logging
# # # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# # # logger = logging.getLogger(__name__)

# # # class LegalSemanticSegmentation(nn.Module):
# # #     """Neural network model for legal document segmentation."""
    
# # #     def __init__(self, bert_model, num_labels=7):
# # #         super(LegalSemanticSegmentation, self).__init__()
# # #         self.bert = bert_model
# # #         self.dropout = nn.Dropout(0.1)
# # #         self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)

# # #     def forward(self, input_ids, attention_mask=None, token_type_ids=None):
# # #         outputs = self.bert(
# # #             input_ids=input_ids, 
# # #             attention_mask=attention_mask, 
# # #             token_type_ids=token_type_ids, 
# # #             return_dict=True
# # #         )
# # #         cls_output = outputs.last_hidden_state[:, 0, :]
# # #         cls_output = self.dropout(cls_output)
# # #         logits = self.classifier(cls_output)
# # #         return logits


# # # class ModelManager:
# # #     """Manager for loading and accessing NLP models on demand."""
    
# # #     def __init__(self, config):
# # #         self.config = config
# # #         self._seg_tokenizer = None
# # #         self._seg_model = None
# # #         self._llm_tokenizer = None
# # #         self._llm_model = None
# # #         self._ner_tokenizer = None
# # #         self._ner_model = None
        
# # #     @property
# # #     def seg_tokenizer(self):
# # #         """Lazy-load the segmentation tokenizer."""
# # #         if self._seg_tokenizer is None:
# # #             logger.info(f"Loading segmentation tokenizer from {self.config.SEGMENTATION_MODEL_PATH}")
# # #             try:
# # #                 self._seg_tokenizer = AutoTokenizer.from_pretrained(self.config.SEGMENTATION_MODEL_PATH)
# # #             except Exception as e:
# # #                 logger.error(f"Failed to load segmentation tokenizer: {e}")
# # #                 raise
# # #         return self._seg_tokenizer
        
# # #     @property
# # #     def seg_model(self):
# # #         """Lazy-load the segmentation model."""
# # #         if self._seg_model is None:
# # #             logger.info(f"Loading segmentation model from {self.config.SEGMENTATION_MODEL_PATH}")
# # #             try:
# # #                 base_model = AutoModel.from_pretrained(self.config.SEGMENTATION_MODEL_PATH)
# # #                 self._seg_model = LegalSemanticSegmentation(base_model).to(self.config.device)
# # #             except Exception as e:
# # #                 logger.error(f"Failed to load segmentation model: {e}")
# # #                 raise
# # #         return self._seg_model
        
# # #     @property
# # #     def llm_tokenizer(self):
# # #         """Lazy-load the LLM tokenizer."""
# # #         if self._llm_tokenizer is None:
# # #             logger.info(f"Loading LLM tokenizer from {self.config.LLM_MODEL_PATH}")
# # #             try:
# # #                 self._llm_tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL_PATH)
# # #                 self._llm_tokenizer.padding_side = 'left'
# # #                 self._llm_tokenizer.pad_token = self._llm_tokenizer.eos_token
# # #             except Exception as e:
# # #                 logger.error(f"Failed to load LLM tokenizer: {e}")
# # #                 raise
# # #         return self._llm_tokenizer
        
# # #     @property
# # #     def llm_model(self):
# # #         """Lazy-load the LLM model with PEFT adapter."""
# # #         if self._llm_model is None:
# # #             logger.info(f"Loading LLM model from {self.config.LLM_MODEL_PATH} with adapter {self.config.LLM_ADAPTER_PATH}")
# # #             try:
# # #                 base_model = AutoModelForCausalLM.from_pretrained(
# # #                     self.config.LLM_MODEL_PATH, 
# # #                     device_map='auto',
# # #                     low_cpu_mem_usage=True
# # #                 )
# # #                 self._llm_model = PeftModel.from_pretrained(
# # #                     base_model, 
# # #                     self.config.LLM_ADAPTER_PATH
# # #                 ).to(self.config.device)
# # #             except Exception as e:
# # #                 logger.error(f"Failed to load LLM model: {e}")
# # #                 raise
# # #         return self._llm_model
        
# # #     @property
# # #     def ner_tokenizer(self):
# # #         """Lazy-load the NER tokenizer."""
# # #         if self._ner_tokenizer is None:
# # #             logger.info(f"Loading NER tokenizer from {self.config.NER_MODEL_PATH}")
# # #             try:
# # #                 self._ner_tokenizer = AutoTokenizer.from_pretrained(self.config.NER_MODEL_PATH)
# # #             except Exception as e:
# # #                 logger.error(f"Failed to load NER tokenizer: {e}")
# # #                 raise
# # #         return self._ner_tokenizer
        
# # #     @property
# # #     def ner_model(self):
# # #         """Lazy-load the NER model."""
# # #         if self._ner_model is None:
# # #             logger.info(f"Loading NER model from {self.config.NER_MODEL_PATH}")
# # #             try:
# # #                 self._ner_model = AutoModelForTokenClassification.from_pretrained(
# # #                     self.config.NER_MODEL_PATH
# # #                 ).to(self.config.device)
# # #             except Exception as e:
# # #                 logger.error(f"Failed to load NER model: {e}")
# # #                 raise
# # #         return self._ner_model
# # #     def validate_device_setup(self):
# # #     logger.info(f"Current device configuration: {self.config.device}")
# # #     if self.config.device == 'mps':
# # #         import torch
# # #         if not hasattr(torch.backends, 'mps'):
# # #             logger.warning("MPS not available in this PyTorch build")
# # #         elif not torch.backends.mps.is_available():
# # #             logger.warning("MPS not available on this hardware")



# # # # Create a global model manager instance
# # # model_manager = ModelManager(config)

# # import torch
# # import torch.nn as nn
# # from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification
# # from peft import PeftModel, PeftConfig
# # from typing import Dict, Any, List, Tuple, Optional
# # import logging
# # from .config import config

# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# # logger = logging.getLogger(__name__)

# # class LegalSemanticSegmentation(nn.Module):
# #     def __init__(self, bert_model, num_labels=7):
# #         super(LegalSemanticSegmentation, self).__init__()
# #         self.bert = bert_model
# #         self.dropout = nn.Dropout(0.1)
# #         self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)

# #     def forward(self, input_ids, attention_mask=None, token_type_ids=None):
# #         outputs = self.bert(
# #             input_ids=input_ids,
# #             attention_mask=attention_mask,
# #             token_type_ids=token_type_ids,
# #             return_dict=True
# #         )
# #         cls_output = outputs.last_hidden_state[:, 0, :]
# #         cls_output = self.dropout(cls_output)
# #         logits = self.classifier(cls_output)
# #         return logits

# # class ModelManager:
# #     def __init__(self, config):
# #         self.config = config
# #         self._seg_tokenizer = None
# #         self._seg_model = None
# #         self._llm_tokenizer = None
# #         self._llm_model = None
# #         self._ner_tokenizer = None
# #         self._ner_model = None

# #     def validate_device_setup(self):
# #         logger.info(f"Current device configuration: {self.config.device}")
# #         if self.config.device == 'mps':
# #             import torch
# #             if not hasattr(torch.backends, 'mps'):
# #                 logger.warning("MPS not available in this PyTorch build")
# #             elif not torch.backends.mps.is_available():
# #                 logger.warning("MPS not available on this hardware")

# #     @property
# #     def seg_tokenizer(self):
# #         if self._seg_tokenizer is None:
# #             logger.info(f'Loading segmentation tokenizer from {self.config.SEGMENTATION_MODEL_PATH}')
# #             try:
# #                 self._seg_tokenizer = AutoTokenizer.from_pretrained(self.config.SEGMENTATION_MODEL_PATH)
# #             except Exception as e:
# #                 logger.error(f'Failed to load segmentation tokenizer: {e}')
# #                 raise
# #         return self._seg_tokenizer

# #     @property
# #     def seg_model(self):
# #         if self._seg_model is None:
# #             logger.info(f'Loading segmentation model from {self.config.SEGMENTATION_MODEL_PATH}')
# #             try:
# #                 base_model = AutoModel.from_pretrained(self.config.SEGMENTATION_MODEL_PATH)
# #                 self._seg_model = LegalSemanticSegmentation(base_model).to(self.config.device)
# #                 self._seg_model.eval()
# #             except Exception as e:
# #                 logger.error(f'Failed to load segmentation model: {e}')
# #                 raise
# #         return self._seg_model

# #     @property
# #     def llm_tokenizer(self):
# #         if self._llm_tokenizer is None:
# #             logger.info(f'Loading LLM tokenizer from {self.config.LLM_MODEL_PATH}')
# #             try:
# #                 self._llm_tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL_PATH)
# #                 self._llm_tokenizer.padding_side = 'left'
# #                 self._llm_tokenizer.pad_token = self._llm_tokenizer.eos_token
# #             except Exception as e:
# #                 logger.error(f'Failed to load LLM tokenizer: {e}')
# #                 raise
# #         return self._llm_tokenizer

# #     @property
# #     def llm_model(self):
# #         if self._llm_model is None:
# #             logger.info(f'Loading LLM model from {self.config.LLM_MODEL_PATH} with adapter {self.config.LLM_ADAPTER_PATH}')
# #             self.validate_device_setup()
            
# #             try:
# #                 # Try loading with GPU/MPS first
# #                 base_model = AutoModelForCausalLM.from_pretrained(
# #                     self.config.LLM_MODEL_PATH,
# #                     device_map=self.config.device_map,
# #                     torch_dtype=torch.float16 if self.config.device != 'cpu' else torch.float32,
# #                     low_cpu_mem_usage=True
# #                 )
                
# #                 self._llm_model = PeftModel.from_pretrained(
# #                     base_model,
# #                     self.config.LLM_ADAPTER_PATH,
# #                     device_map=self.config.device_map
# #                 )
# #                 self._llm_model.eval()
                
# #                 if self.config.device == 'mps':
# #                     import torch.mps
# #                     torch.mps.empty_cache()
                    
# #             except Exception as e:
# #                 logger.error(f'Primary loading failed: {e}. Attempting CPU fallback...')
# #                 try:
# #                     base_model = AutoModelForCausalLM.from_pretrained(
# #                         self.config.LLM_MODEL_PATH,
# #                         device_map='cpu',
# #                         torch_dtype=torch.float32
# #                     )
# #                     self._llm_model = PeftModel.from_pretrained(
# #                         base_model,
# #                         self.config.LLM_ADAPTER_PATH,
# #                         device_map='cpu'
# #                     )
# #                     self._llm_model.eval()
# #                 except Exception as fallback_e:
# #                     logger.error(f'CPU fallback failed: {fallback_e}')
# #                     raise
# #         return self._llm_model

# #     @property
# #     def ner_tokenizer(self):
# #         if self._ner_tokenizer is None:
# #             logger.info(f'Loading NER tokenizer from {self.config.NER_MODEL_PATH}')
# #             try:
# #                 self._ner_tokenizer = AutoTokenizer.from_pretrained(self.config.NER_MODEL_PATH)
# #             except Exception as e:
# #                 logger.error(f'Failed to load NER tokenizer: {e}')
# #                 raise
# #         return self._ner_tokenizer

# #     @property
# #     def ner_model(self):
# #         if self._ner_model is None:
# #             logger.info(f'Loading NER model from {self.config.NER_MODEL_PATH}')
# #             try:
# #                 self._ner_model = AutoModelForTokenClassification.from_pretrained(
# #                     self.config.NER_MODEL_PATH
# #                 ).to(self.config.device)
# #                 self._ner_model.eval()
# #             except Exception as e:
# #                 logger.error(f'Failed to load NER model: {e}')
# #                 raise
# #         return self._ner_model

# # model_manager = ModelManager(config)

# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification
# from peft import PeftModel, PeftConfig
# from typing import Dict, Any, List, Tuple, Optional
# import logging
# from .config import config

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class LegalSemanticSegmentation(nn.Module):
#     def __init__(self, bert_model, num_labels=7):
#         super(LegalSemanticSegmentation, self).__init__()
#         self.bert = bert_model
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)

#     def forward(self, input_ids, attention_mask=None, token_type_ids=None):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             return_dict=True
#         )
#         cls_output = outputs.last_hidden_state[:, 0, :]
#         cls_output = self.dropout(cls_output)
#         logits = self.classifier(cls_output)
#         return logits

# class ModelManager:
#     def __init__(self, config):
#         self.config = config
#         self._seg_tokenizer = None
#         self._seg_model = None
#         self._llm_tokenizer = None
#         self._llm_model = None
#         self._ner_tokenizer = None
#         self._ner_model = None

#     def validate_device_setup(self):
#         logger.info(f"Current device configuration: {self.config.device}")
#         if self.config.device == 'mps':
#             if not hasattr(torch.backends, 'mps'):
#                 logger.warning("MPS not available in this PyTorch build")
#             elif not torch.backends.mps.is_available():
#                 logger.warning("MPS not available on this hardware")

#     @property
#     def seg_tokenizer(self):
#         if self._seg_tokenizer is None:
#             logger.info(f'Loading segmentation tokenizer from {self.config.SEGMENTATION_MODEL_PATH}')
#             try:
#                 self._seg_tokenizer = AutoTokenizer.from_pretrained(self.config.SEGMENTATION_MODEL_PATH)
#             except Exception as e:
#                 logger.error(f'Failed to load segmentation tokenizer: {e}')
#                 raise
#         return self._seg_tokenizer

#     @property
#     def seg_model(self):
#         if self._seg_model is None:
#             logger.info(f'Loading segmentation model from {self.config.SEGMENTATION_MODEL_PATH}')
#             try:
#                 base_model = AutoModel.from_pretrained(self.config.SEGMENTATION_MODEL_PATH)
#                 self._seg_model = LegalSemanticSegmentation(base_model).to(self.config.device)
#                 self._seg_model.eval()
#             except Exception as e:
#                 logger.error(f'Failed to load segmentation model: {e}')
#                 raise
#         return self._seg_model

#     @property
#     def llm_tokenizer(self):
#         if self._llm_tokenizer is None:
#             logger.info(f'Loading LLM tokenizer from {self.config.LLM_MODEL_PATH}')
#             try:
#                 self._llm_tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL_PATH)
#                 self._llm_tokenizer.padding_side = 'left'
#                 self._llm_tokenizer.pad_token = self._llm_tokenizer.eos_token
#             except Exception as e:
#                 logger.error(f'Failed to load LLM tokenizer: {e}')
#                 raise
#         return self._llm_tokenizer

#     @property
#     def llm_model(self):
#         if self._llm_model is None:
#             logger.info(f'Loading LLM model from {self.config.LLM_MODEL_PATH}')
#             self.validate_device_setup()
#             try:
#                 self._llm_model = AutoModelForCausalLM.from_pretrained(
#                     self.config.LLM_MODEL_PATH,
#                     device_map=self.config.device_map,
#                     torch_dtype=torch.float16 if self.config.device != 'cpu' else torch.float32,
#                     low_cpu_mem_usage=True
#                 )
#                 self._llm_model.eval()
#                 if self.config.device == 'mps':
#                     torch.mps.empty_cache()
#             except Exception as e:
#                 logger.error(f'Primary loading failed: {e}. Attempting CPU fallback...')
#                 try:
#                     self._llm_model = AutoModelForCausalLM.from_pretrained(
#                         self.config.LLM_MODEL_PATH,
#                         device_map='cpu',
#                         torch_dtype=torch.float32
#                     )
#                     self._llm_model.eval()
#                 except Exception as fallback_e:
#                     logger.error(f'CPU fallback failed: {fallback_e}')
#                     raise RuntimeError(f'Failed to load model: {fallback_e}')
#         return self._llm_model
#     @property
#     def ner_tokenizer(self):
#         if self._ner_tokenizer is None:
#             logger.info(f'Loading NER tokenizer from {self.config.NER_MODEL_PATH}')
#             try:
#                 self._ner_tokenizer = AutoTokenizer.from_pretrained(self.config.NER_MODEL_PATH)
#             except Exception as e:
#                 logger.error(f'Failed to load NER tokenizer: {e}')
#                 raise
#         return self._ner_tokenizer

#     @property
#     def ner_model(self):
#         if self._ner_model is None:
#             logger.info(f'Loading NER model from {self.config.NER_MODEL_PATH}')
#             try:
#                 self._ner_model = AutoModelForTokenClassification.from_pretrained(
#                     self.config.NER_MODEL_PATH
#                 ).to(self.config.device)
#                 self._ner_model.eval()
#             except Exception as e:
#                 logger.error(f'Failed to load NER model: {e}')
#                 raise
#         return self._ner_model

# model_manager = ModelManager(config)

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification
from peft import PeftModel, PeftConfig
from typing import Dict, Any, List, Tuple, Optional
import logging
from .config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalSemanticSegmentation(nn.Module):
    def __init__(self, bert_model, num_labels=7):
        super(LegalSemanticSegmentation, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

class ModelManager:
    def __init__(self, config):
        self.config = config
        self._seg_tokenizer = None
        self._seg_model = None
        self._llm_tokenizer = None
        self._llm_model = None
        self._ner_tokenizer = None
        self._ner_model = None

    def validate_device_setup(self):
        logger.info(f"Current device configuration: {self.config.device}")
        if self.config.device == 'mps':
            if not hasattr(torch.backends, 'mps'):
                logger.warning("MPS not available in this PyTorch build")
            elif not torch.backends.mps.is_available():
                logger.warning("MPS not available on this hardware")

    @property
    def seg_tokenizer(self):
        if self._seg_tokenizer is None:
            logger.info(f'Loading segmentation tokenizer from {self.config.SEGMENTATION_MODEL_PATH}')
            try:
                self._seg_tokenizer = AutoTokenizer.from_pretrained(self.config.SEGMENTATION_MODEL_PATH)
            except Exception as e:
                logger.error(f'Failed to load segmentation tokenizer: {e}')
                raise
        return self._seg_tokenizer

    @property
    def seg_model(self):
        if self._seg_model is None:
            logger.info(f'Loading segmentation model from {self.config.SEGMENTATION_MODEL_PATH}')
            try:
                base_model = AutoModel.from_pretrained(self.config.SEGMENTATION_MODEL_PATH)
                self._seg_model = LegalSemanticSegmentation(base_model).to(self.config.device)
                self._seg_model.eval()
            except Exception as e:
                logger.error(f'Failed to load segmentation model: {e}')
                raise
        return self._seg_model

    @property
    def llm_tokenizer(self):
        if self._llm_tokenizer is None:
            logger.info(f'Loading LLM tokenizer from {self.config.LLM_MODEL_PATH}')
            try:
                self._llm_tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL_PATH)
                self._llm_tokenizer.padding_side = 'left'
                self._llm_tokenizer.pad_token = self._llm_tokenizer.eos_token
            except Exception as e:
                logger.error(f'Failed to load LLM tokenizer: {e}')
                raise
        return self._llm_tokenizer

    @property
    def llm_model(self):
        if self._llm_model is None:
            logger.info(f'Loading LLM model from {self.config.LLM_MODEL_PATH}')
            self.validate_device_setup()
            try:
                if self.config.device != 'cpu':
                    self._llm_model = AutoModelForCausalLM.from_pretrained(
                        self.config.LLM_MODEL_PATH,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        offload_folder="./offload"
                    )
                else:
                    raise RuntimeError("Skipping MPS for now")
    
            except Exception as e:
                logger.error(f'Primary loading failed: {e}. Attempting CPU fallback...')
                try:
                    self._llm_model = AutoModelForCausalLM.from_pretrained(
                        self.config.LLM_MODEL_PATH,
                        device_map='cpu',
                        torch_dtype=torch.float32
                    )
                    self._llm_model.eval()
                except Exception as fallback_e:
                    logger.error(f'CPU fallback failed: {fallback_e}')
                    raise RuntimeError(f'Failed to load model: {fallback_e}')
        return self._llm_model
    @property
    def ner_tokenizer(self):
        if self._ner_tokenizer is None:
            logger.info(f'Loading NER tokenizer from {self.config.NER_MODEL_PATH}')
            try:
                self._ner_tokenizer = AutoTokenizer.from_pretrained(self.config.NER_MODEL_PATH)
            except Exception as e:
                logger.error(f'Failed to load NER tokenizer: {e}')
                raise
        return self._ner_tokenizer

    @property
    def ner_model(self):
        if self._ner_model is None:
            logger.info(f'Loading NER model from {self.config.NER_MODEL_PATH}')
            try:
                self._ner_model = AutoModelForTokenClassification.from_pretrained(
                    self.config.NER_MODEL_PATH
                ).to(self.config.device)
                self._ner_model.eval()
            except Exception as e:
                logger.error(f'Failed to load NER model: {e}')
                raise
        return self._ner_model

model_manager = ModelManager(config)