# Installing GNAT and GNATprove, and Lean

Tested on macOS 26 / x86_64 on 2026-10-22. Both toolchains install without an account.

## SPARK (GNATprove) via Alire

```bash
# 1. Alire (the Ada package manager). Download alr-<version>-bin-x86_64-macos.zip from
#    https://github.com/alire-project/alire/releases, unzip, and put bin/alr on your PATH.
alr -n index --update-all
alr -n toolchain --select gnat_native gprbuild   # GNAT 16.1 + gprbuild
alr -n install gnatprove                          # GNATprove 16.1.0 (hash-pinned binary)

# 2. Put the tools on PATH (Alire prints the exact toolchain directory):
export PATH="$HOME/.alire/bin:$(ls -d ~/.local/share/alire/toolchains/gnat_native_*)/bin:$(ls -d ~/.local/share/alire/toolchains/gprbuild_*)/bin:$PATH"

# 3. Prove
make -C spark-llama-safety prove
```

Caveat: the Alire GNATprove is built from an intermediate SPARK Pro commit, so it is not a specific certified release. Qualification for certification evidence would use AdaCore's qualified product.

## Lean 4

```bash
brew install elan-init     # compiles coreutils from source on some macOS versions; can take several minutes
cd lean-kernel && lake build
```

The toolchain version is pinned in `lean-kernel/lean-toolchain`.
