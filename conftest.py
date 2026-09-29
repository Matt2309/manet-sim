# File vuoto: la sua sola presenza nella root del repository fa sì che pytest
# aggiunga questa directory a sys.path, rendendo importabile il pacchetto `src`
# dai test in tests/test_mobility.py (import src.mobility).
