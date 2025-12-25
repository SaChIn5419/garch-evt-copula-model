import sys
import time

# Import modules
# We wrap in try-except to ensure the pipeline doesn't crash effectively on simple import errors
try:
    import quant_model
    import vine_stress_test
except ImportError as e:
    print(f"[FATAL ERROR] Could not import required modules: {e}")
    print("Ensure 'quant_model.py' and 'vine_stress_test.py' are in the same folder.")
    sys.exit(1)

def print_header(title):
    print("\n" + "#" * 80)
    print(f"   {title}")
    print("#" * 80 + "\n")

def run_pipeline():
    print_header("STARTING ADVANCED QUANTITATIVE PIPELINE")
    print("Modules Loaded Successfully.")
    
    # ----------------------------------------------------
    # PHASE 1: GLOBAL MACRO & HEDGING ANALYSIS
    # ----------------------------------------------------
    print_header("PHASE 1: GLOBAL PAIR ANALYSIS (GARCH-COPULA)")
    print(">> Objective: Optimize strategic pairs for Capital Preservation & STARR Ratio.")
    print(">> Model: GARCH(1,1)-t + Student-t Copula\n")
    
    try:
        quant_model.main()
    except Exception as e:
        print(f"[ERROR] Phase 1 Failed: {e}")

    # ----------------------------------------------------
    # PHASE 2: STRUCTURAL ADAPTATION & STRESS TESTING
    # ----------------------------------------------------
    print_header("PHASE 2: STRUCTURAL VINE STRESS TESTING")
    print(">> Objective: Diagnose Market Structure (C-Vine vs D-Vine) in Crisis Clusters.")
    print(">> Feature: 'Crisis vs Now' Backtesting (COVID vs Today).\n")
    
    try:
        vine_stress_test.run_batch_stress_test()
    except Exception as e:
        print(f"[ERROR] Phase 2 Failed: {e}")

    print_header("PIPELINE EXECUTION COMPLETE")
    print("Results have been printed above.")

if __name__ == "__main__":
    run_pipeline()
    input("\nPress Enter to exit...")
