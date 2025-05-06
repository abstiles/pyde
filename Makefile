.PHONY: depends package

PACKAGE := pyde-ssg

package:
	python -m build

depends:
	mkdir -p _build/diff-dom
	curl -o _build/diff-dom.tgz $(shell curl -s https://registry.npmjs.org/diff-dom/latest | jq -r '.dist.tarball')
	mkdir -p pyde/js/vendored/diff-dom
	tar xvf _build/diff-dom.tgz -C _build/diff-dom
	cp -r _build/diff-dom/package/dist/* pyde/js/vendored/diff-dom/
	cp _build/diff-dom/package/LICENSE* pyde/js/vendored/diff-dom/
