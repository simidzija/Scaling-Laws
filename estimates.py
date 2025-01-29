# Standard library
from datetime import datetime


################################  GPU  #####################################

# Date instantiated
DATE_INSTANTIATED = datetime(2025, 1, 24)

# Elapsed time (s)
elapsed_time = (datetime.now() - DATE_INSTANTIATED).total_seconds()

# TFLOPs/s for T4 GPU
TFLOPS = 2  

# Cost of T4 GPU ($/hour)
COST_PER_HOUR = 0.3

# Cost per second ($/s)
COST_PER_SEC = COST_PER_HOUR / (60 * 60)

# Total credits ($)
TOTAL_CREDITS = 300

# Credits remaining ($)
credits_remaining = TOTAL_CREDITS - COST_PER_SEC * elapsed_time

# Time remaining (s)
time_remaining_secs = credits_remaining / COST_PER_SEC

# Time remaining (days)
time_remaining_days = time_remaining_secs / (60 * 60 * 24)

print('----------------------  GPU Estimates  -------------------------')
print(f'Credits remaining: ${credits_remaining:.2f}')
print(f'Time before credits run out: {time_remaining_days:.1f} days')
print('\n')


#############################  Data  ###################################

# Words / MB in English language text
WORDS_PER_MB = 175,000  

# Tokens per word in English (using BPE)
TOKENS_PER_WORD = 1.3  








