# Makefile for Sphinx documentation
#

# Variables
SPHINXBUILD = sphinx-build
SOURCEDIR   = source
BUILDDIR    = build/html

.PHONY: html clean

# Build HTML documentation
html:
	@echo "Building HTML documentation..."
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)"
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/"

# Clean build directory
clean:
	@echo "Cleaning documentation build..."
	rm -rf "$(BUILDDIR)"
	@echo "Clean complete."
