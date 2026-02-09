CC = gcc
CFLAGS = -O2 -Wall -Wextra -Ilibs/include
LDFLAGS = -Llibs/lib -Wl,-rpath,'$$ORIGIN/libs/lib'
LIBS = -lsherpa-onnx-c-api -lportaudio -lpthread

dictate: dictate.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f dictate

.PHONY: clean
