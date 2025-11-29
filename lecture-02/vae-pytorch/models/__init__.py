from .base import *
from .vae import *
from .hvae import *

# Aliases
VAE = VanillaVAE
HVAE = HVAE
vae_models = {'VanillaVAE':VanillaVAE, 
              'HVAE':HVAE
             }
