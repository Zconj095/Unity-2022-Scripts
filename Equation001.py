import numpy as np

def self_defined_memory_retrieval(cdt, umn, cr, sci, f_cdt_func, dot_product_func):
    """
    Calculates the Self-Defined Memory Retrieval (SDMR) score based on the given parameters and user-defined functions.

    Args:
        cdt: A numerical value representing the influence of Created Dictionary Terminology (CDT) on retrieval.
        umn: A numerical value representing the Utilization of Memory Management Notes (UMN).
        cr: A numerical value representing the Comprehension of Bodily Effects (CR).
        sci: A numerical value representing the Self-Defining Critical Information (SCI).
        f_cdt_func: A function representing the influence of CDT on retrieval.
        dot_product_func: A function taking UMN, CR, and SCI as inputs and returning their weighted dot product.

    Returns:
        A numerical value representing the overall SDMR score.
    """

  # Apply user-defined function for CDT influence
    f_cdt = f_cdt_func(cdt)

  # Calculate weighted dot product using user-defined function
    dot_product = dot_product_func(umn, cr, sci)

  # Calculate SDMR score
    sdmr = f_cdt * dot_product

    return sdmr

# Example usage with custom functions

# Define a custom function for f(CDT) (e.g., exponential)
def custom_f_cdt(cdt):
    return np.exp(cdt)

# Define a custom function for dot product with weights (e.g., UMN weighted more)
def custom_dot_product(umn, cr, sci):
    return 2 * umn * cr + sci

# Use custom functions in SDMR calculation
cdt = 5
umn = 0.8
cr = 0.7
sci = 0.9

sdmr_score = self_defined_memory_retrieval(cdt, umn, cr, sci, custom_f_cdt, custom_dot_product)

print(f"Self-Defined Memory Retrieval (SDMR) score with custom functions: {sdmr_score}")