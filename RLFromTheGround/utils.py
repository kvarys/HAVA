import numpy as np

def format_scientific(number_str):
    number = float(number_str)
    if number == 0:
        return "0e0"  # Handle zero specially

    # Get the exponent
    exponent = int(np.floor(np.log10(abs(number))))

    # Get the significant digits
    significant_digits = number / (10 ** exponent)

    # Format based on the value of significant digits
    if significant_digits == 1:
        formatted_number = f"1e{exponent}"
    else:
        # Remove trailing decimal point and zeros if they exist
        significant_str = f"{significant_digits:.15f}".rstrip('0').rstrip('.')
        # Replace the dot with an underscore
        significant_str = significant_str.replace('.', '_')
        formatted_number = f"{significant_str}e{exponent}"

    return formatted_number.replace("-","")

def non_default_args(args, parser):
    result = []
    for arg in vars(args):
        user_val = getattr(args, arg)
        default_val = parser.get_default(arg)
        if user_val != default_val and arg != "game" and arg != "agent_name":
            if arg=="lr":
                user_val = format_scientific(user_val)

            result.append(f"{arg}_{user_val}")

    result.append(f"game_{args.game}")
    return '_'.join(result)
