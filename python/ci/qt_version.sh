# the Qt of the wheel: the minimum PySide6 of pyproject.toml (e.g. "PySide6>=6.12" -> 6.12.0), or NEUTRINO_QT_VERSION.
# Qt is backward compatible: compiled with Qt 6.12 the module works with the Qt of any PySide6 >= 6.12
PYPROJECT="$(dirname "${BASH_SOURCE[0]}")/../../pyproject.toml"
QT_VERSION=${NEUTRINO_QT_VERSION:-$(sed -n 's/.*"PySide6>=\([0-9]*\.[0-9]*\).*/\1.0/p' "$PYPROJECT")}
echo "Qt for the wheel: $QT_VERSION"
