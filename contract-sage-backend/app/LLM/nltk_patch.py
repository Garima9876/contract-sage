import os
import nltk
import logging
import importlib
from nltk.data import find, load
from nltk.tag import _POS_TAGGER
logger = logging.getLogger(__name__)

def apply_nltk_patches():
    """Apply comprehensive patches to fix NLTK resource loading issues"""
    # First, create the nltk_data directory if it doesn't exist
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Download resources with explicit destination
    try:
        nltk.download('averaged_perceptron_tagger', quiet=True, download_dir=nltk_data_dir)
        nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)
        nltk.download('wordnet', quiet=True, download_dir=nltk_data_dir)
    except Exception as e:
        logger.warning(f'Failed to download NLTK resources: {e}')
    
    # Important: Modify the POS_TAGGER value in NLTK itself
    # This is what's causing the '_eng' suffix issue
    if hasattr(nltk.tag, '_POS_TAGGER'):
        if nltk.tag._POS_TAGGER == 'taggers/averaged_perceptron_tagger_eng':
            logger.info('Patching NLTK POS_TAGGER path')
            nltk.tag._POS_TAGGER = 'taggers/averaged_perceptron_tagger'
    
    # Also patch the internal resource loader
    original_load = nltk.data.load

    def patched_load(resource_url, format='auto', **kwargs):
        if 'averaged_perceptron_tagger_eng' in resource_url:
            resource_url = resource_url.replace('averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger')
            logger.info(f'Patched resource URL to: {resource_url}')
        return original_load(resource_url, format, **kwargs)
    
    nltk.data.load = patched_load
    
    # Verify tagger exists
    try:
        tagger_path = os.path.join(nltk_data_dir, 'taggers', 'averaged_perceptron_tagger')
        if os.path.exists(tagger_path):
            logger.info(f'Verified tagger exists at: {tagger_path}')
        else:
            logger.warning(f'Tagger not found at: {tagger_path}')
    except Exception as e:
        logger.warning(f'Error verifying tagger path: {e}')
    
    logger.info('Applied comprehensive NLTK patches')

# Add a direct fix for the POS tag function
def fixed_pos_tag(tokens):
    """A wrapper for NLTK's pos_tag that handles the _eng suffix issue"""
    try:
        return nltk.tag.pos_tag(tokens)
    except LookupError as e:
        if 'averaged_perceptron_tagger_eng' in str(e):
            # Modify the internal constant temporarily
            original = nltk.tag._POS_TAGGER
            nltk.tag._POS_TAGGER = 'taggers/averaged_perceptron_tagger'
            try:
                return nltk.tag.pos_tag(tokens)
            finally:
                nltk.tag._POS_TAGGER = original
        else:
            raise