.PHONY: all books mistral_local

all: books mistral_local

books:
	-mkdir data data/books
	wget -O data/books/frankenstein.txt \
		https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/frankenstein.txt
	wget -O data/books/mobydick.txt \
		https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/mobydick.txt
	wget -O data/books/prideandprejudice.txt \
		https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/prideandprejudice.txt

mistral_local:
	-mkdir models
	# see alternatives: https://www.baeldung.com/cs/hugging-face-model-download
	hf download mistralai/Mistral-7B-v0.1 --cache-dir models

bookreport:
	-mkdir output
	python main.py \
		--input=data/books/frankenstein.txt \
		--delimiter='CHAPTER'\
		--datatype='Event'\
		--outdir=output

# done.
