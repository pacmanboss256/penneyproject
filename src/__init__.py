# Package marker file.
#
# The project historically relied on PEP 420 namespace-package behavior for `src`.
# Creating an explicit package makes builds/backends less fragile while keeping
# runtime imports like `from src.parser import Parser` working the same.

