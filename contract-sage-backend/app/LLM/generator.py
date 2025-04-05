# # # # generator.py
# # # import torch
# # # from typing import Dict, Any
# # # import logging

# # # logger = logging.getLogger(__name__)

# # # def generate_summary(prompt: str) -> str:
# # #     """Generate a legal document summary using the LLM.
    
# # #     Args:
# # #         prompt: The assembled prompt for the LLM
        
# # #     Returns:
# # #         Generated summary text
# # #     """
# # #     from config import config
# # #     from model import model_manager
    
# # #     logger.info("Generating summary with LLM")
    
# # #     try:
# # #         # Tokenize the prompt
# # #         inputs = model_manager.llm_tokenizer(
# # #             prompt, 
# # #             return_tensors='pt', 
# # #             padding=True
# # #         ).to(config.device)
        
# # #         # Generate text
# # #         with torch.no_grad():
# # #             output = model_manager.llm_model.generate(
# # #                 **inputs, 
# # #                 max_new_tokens=config.MAX_TOKENS,
# # #                 do_sample=True, 
# # #                 temperature=config.TEMPERATURE,
# # #                 top_p=0.95,
# # #                 top_k=50,
# # #                 repetition_penalty=1.1
# # #             )
        
# # #         # Decode output
# # #         summary = model_manager.llm_tokenizer.decode(output[0], skip_special_tokens=True)
        
# # #         # Extract just the generated part (after the prompt)
# # #         if prompt in summary:
# # #             summary = summary[len(prompt):].strip()
        
# # #         logger.info(f"Summary generation complete: {len(summary)} characters")
# # #         return summary
        
# # #     except Exception as e:
# # #         logger.error(f"Error generating summary: {e}")
# # #         return "Error generating summary. Please try again."

# # import torch
# # from typing import Dict, Any
# # import logging
# # logger = logging.getLogger(__name__)

# # def generate_summary(prompt: str) -> str:
# #     from config import config
# #     from model import model_manager
    
# #     logger.info('Generating summary with LLM')
# #     try:
# #         device = config.device
        
# #         # Clear cache if using MPS
# #         if device == 'mps':
# #             import torch.mps
# #             torch.mps.empty_cache()
            
# #         # Tokenize input
# #         inputs = model_manager.llm_tokenizer(
# #             prompt,
# #             return_tensors='pt',
# #             padding=True,
# #             truncation=True,
# #             max_length=2048
# #         ).to(device)
        
# #         # Prepare generation config
# #         generation_config = {
# #             'max_new_tokens': config.MAX_TOKENS,
# #             'do_sample': True,
# #             'temperature': config.TEMPERATURE,
# #             'top_p': 0.95,
# #             'top_k': 50,
# #             'repetition_penalty': 1.1,
# #             'pad_token_id': model_manager.llm_tokenizer.eos_token_id
# #         }
        
# #         # Generate with appropriate device handling
# #         with torch.no_grad():
# #             if device == 'mps':
# #                 with torch.autocast('mps'):
# #                     output = model_manager.llm_model.generate(
# #                         **inputs,
# #                         **generation_config
# #                     )
# #             else:
# #                 output = model_manager.llm_model.generate(
# #                     **inputs,
# #                     **generation_config
# #                 )
                
# #         # Decode and clean output
# #         summary = model_manager.llm_tokenizer.decode(
# #             output[0],
# #             skip_special_tokens=True
# #         )
        
# #         if prompt in summary:
# #             summary = summary[len(prompt):].strip()
            
# #         logger.info(f'Summary generation complete: {len(summary)} characters')
# #         return summary
        
# #     except RuntimeError as e:
# #         if 'MPS' in str(e):
# #             logger.warning('MPS error occurred, retrying with CPU')
# #             try:
# #                 # Move model and inputs to CPU
# #                 model_manager._llm_model = model_manager._llm_model.to('cpu')
# #                 inputs = inputs.to('cpu')
                
# #                 with torch.no_grad():
# #                     output = model_manager.llm_model.generate(
# #                         **inputs,
# #                         **generation_config
# #                     )
# #                 summary = model_manager.llm_tokenizer.decode(
# #                     output[0],
# #                     skip_special_tokens=True
# #                 )
# #                 return summary
# #             except Exception as cpu_e:
# #                 logger.error(f'CPU fallback failed: {cpu_e}')
# #                 return 'Error generating summary. Please try again.'
        
# #         logger.error(f'Error generating summary: {e}')
# #         return f'Error generating summary: {str(e)}'
    
# #     except Exception as e:
# #         logger.error(f'Unexpected error in generation: {e}')
# #         return 'Error generating summary. Please try again.'

# import torch
# from typing import Dict, Any
# import logging
# logger = logging.getLogger(__name__)

# def generate_summary(prompt: str) -> str:
#     from config import config
#     from model import model_manager
#     logger.info('Generating summary with LLM')
#     try:
#         device = config.device
#         if device == 'mps':
#             torch.mps.empty_cache()
        
#         inputs = model_manager.llm_tokenizer(
#             prompt, 
#             return_tensors='pt', 
#             padding=True, 
#             truncation=True, 
#             max_length=2048
#         ).to(device)
        
#         generation_config = {
#             'max_new_tokens': config.MAX_TOKENS,
#             'do_sample': True,
#             'temperature': config.TEMPERATURE,
#             'top_p': 0.9,
#             'top_k': 50,
#             'repetition_penalty': 1.1,
#             'pad_token_id': model_manager.llm_tokenizer.eos_token_id
#         }
        
#         with torch.no_grad():
#             if device == 'mps':
#                 with torch.autocast('mps'):
#                     output = model_manager.llm_model.generate(**inputs, **generation_config)
#             else:
#                 output = model_manager.llm_model.generate(**inputs, **generation_config)
        
#         summary = model_manager.llm_tokenizer.decode(output[0], skip_special_tokens=True)
        
#         if prompt in summary:
#             summary = summary[len(prompt):].strip()
        
#         logger.info(f'Summary generation complete: {len(summary)} characters')
#         return summary
#     # ... rest of the error handling remains the same ...
        
#     except RuntimeError as e:
#         if 'MPS' in str(e):
#             logger.warning('MPS error occurred, retrying with CPU')
#             try:
#                 model_manager._llm_model = model_manager._llm_model.to('cpu')
#                 inputs = inputs.to('cpu')
#                 with torch.no_grad():
#                     output = model_manager.llm_model.generate(
#                         **inputs,
#                         **generation_config
#                     )
#                 summary = model_manager.llm_tokenizer.decode(
#                     output[0],
#                     skip_special_tokens=True
#                 )
#                 return summary
#             except Exception as cpu_e:
#                 logger.error(f'CPU fallback failed: {cpu_e}')
#                 return 'Error generating summary. Please try again.'
        
#         logger.error(f'Error generating summary: {e}')
#         return f'Error generating summary: {str(e)}'
    
#     except Exception as e:
#         logger.error(f'Unexpected error in generation: {e}')
#         return 'Error generating summary. Please try again.'

import torch
from typing import Dict, Any
import logging
from config import config
from model import model_manager

logger = logging.getLogger(__name__)

def generate_summary(prompt: str) -> str:
    logger.info('Generating summary with LLM')
    try:
        # Ensure model and inputs are on same device
        device = model_manager.llm_model.device
        
        inputs = model_manager.llm_tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=config.MAX_TOKENS
        ).to(device)

        generation_config = {
            'max_new_tokens': config.MAX_TOKENS,
            'do_sample': True,
            'temperature': config.TEMPERATURE,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.1,
            'pad_token_id': model_manager.llm_tokenizer.eos_token_id
        }

        with torch.no_grad():
            if config.device == 'mps':
                with torch.autocast('mps'):
                    output = model_manager.llm_model.generate(**inputs, **generation_config)
            else:
                output = model_manager.llm_model.generate(**inputs, **generation_config)

        summary = model_manager.llm_tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Clean up output
        if prompt in summary:
            summary = summary[len(prompt):].strip()
        
        logger.info(f'Generated summary with {len(summary)} characters')
        return summary

    except RuntimeError as e:
        if 'MPS' in str(e):
            logger.warning('MPS error, retrying with CPU')
            try:
                inputs = inputs.to('cpu')
                model_manager.llm_model = model_manager.llm_model.to('cpu')
                with torch.no_grad():
                    output = model_manager.llm_model.generate(**inputs, **generation_config)
                return model_manager.llm_tokenizer.decode(output[0], skip_special_tokens=True)
            except Exception as cpu_e:
                logger.error(f'CPU fallback failed: {cpu_e}')
                return 'Generation error: Please try again with shorter text'

        logger.error(f'Generation error: {e}')
        return f'Generation failed: {str(e)}'

    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        return 'An error occurred during generation'