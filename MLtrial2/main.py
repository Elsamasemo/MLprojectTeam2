import sys
from contextlib import redirect_stdout
from src.baseline import run_baseline
from src.config import OUT_DIR
from src.load import load_raw
from src.audit import run_audit, tidy
from src.plots import run_plots
from src.clean import clean_base, validate_clean
from src.split import prepare_data
from src.variants import learn_variant_params, make_variants, variant_summary
from src.modeling import compare_variants


class Tee:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def build_train_variants():

    train_base, _test_base, params, _log = prepare_data()
    params = learn_variant_params(train_base, params)
    return make_variants(train_base, params)


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "audit"
    OUT_DIR.mkdir(exist_ok=True)

    if command == "audit":
        with open(OUT_DIR / "audit_report.txt", "w", encoding="utf-8") as f:
            with redirect_stdout(Tee(sys.stdout, f)):
                run_audit(load_raw())
    elif command == "plots":
        run_plots(tidy(load_raw()))
    elif command == "clean":
        base, params, log = clean_base(load_raw(), is_train=True)
        print("\n".join(log))
        print("Problems found:", validate_clean(base) or "none")
        print("Shape:", base.shape)
    elif command == "split":
        train_base, test_base, params, log = prepare_data()
        print("\n".join(log))
        print("Train:", train_base.shape, "| Test:", test_base.shape)
        print("Problems (train):", validate_clean(train_base) or "none")
        print("Problems (test) :", validate_clean(test_base) or "none")
        (OUT_DIR / "cleaning_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    elif command == "variants":
        print(variant_summary(build_train_variants()).to_string(index=False))
    elif command == "compare":
        compare_variants(build_train_variants())
    elif command == "baseline":
        run_baseline()
    else:
        sys.exit(f"Unknown command '{command}'. Use: audit | plots | clean | split | variants | compare | baseline")


if __name__ == "__main__":
    main()