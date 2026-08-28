bot_dataset:
	python3 audio_digits_dataset_bot.py
bot_recognition:
	python3 audio_digits_recognition_bot.py	
vad_sort:
	{ \
	for folder in 0 1 2 3 4 5 6 7 8 9; \
	do \
		mkdir -p dataset/splitted/$$folder; \
	done \
	}
	{ \
	for fname in $$(ls dataset/wav/* | grep wav); \
	do \
		python3 split_by_vad.py $$fname 0.1 0.005 dataset/splitted; \
	done \
	} 
splitted:
	cp -r dataset/splitted/* dataset/splitted_final/
	rm -f dataset/splitted/*/*
	rm -f dataset/wav/*
	rm -f dataset/ogg/*
training:
	python3 training.py