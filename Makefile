.PHONY: clean
clean:
	rm -rf python/__pycache__ python/*/__pycache__
	rm -rf python/*.pyc python/*/*.pyc
	rm -f cpp/*.so
