# generator.py
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def generate_summary(prompt: str) -> str:
    """Generate a legal document summary using the LLM.
    
    Args:
        prompt: The assembled prompt for the LLM
        
    Returns:
        Generated summary text
    """
    from config import config
    from model import model_manager
    
    logger.info("Generating summary with LLM")
    
    try:
        # Tokenize the prompt
        inputs = model_manager.llm_tokenizer(
            prompt, 
            return_tensors='pt', 
            padding=True
        ).to(config.device)
        
        # Generate text
        with torch.no_grad():
            output = model_manager.llm_model.generate(
                **inputs, 
                max_new_tokens=config.MAX_TOKENS,
                do_sample=True, 
                temperature=config.TEMPERATURE,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1
            )
        
        # Decode output
        summary = model_manager.llm_tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the generated part (after the prompt)
        if prompt in summary:
            summary = summary[len(prompt):].strip()
        
        logger.info(f"Summary generation complete: {len(summary)} characters")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return "Error generating summary. Please try again."
