.PHONY: clean all

all: remdi.js il.js 407.js cunning-fox.js 257.js 111.js

remdi.js: remdi.cc /tmp/gen.cc
	/media/big/code-external/emsdk_portable/emscripten/master/em++ -std=c++11 -g -O3 remdi.cc -o remdi.js
il.js: il.cc /tmp/gen.cc
	/media/big/code-external/emsdk_portable/emscripten/master/em++ -std=c++11 -O1 il.cc -o il.js
present.js: present.cc /tmp/gen.cc
	/media/big/code-external/emsdk_portable/emscripten/master/em++ -std=c++11 -O1 present.cc -o present.js
407.js: 407.cc /tmp/gen.cc
	/media/big/code-external/emsdk_portable/emscripten/master/em++ -std=c++11 -O1 407.cc -o 407.js
/tmp/gen.cc: gen-407.py
	python gen-407.py
cunning-fox.js: cunning-fox.cc
	python gen-aa.py
	/media/big/code-external/emsdk_portable/emscripten/master/em++ -std=c++11 -O1 cunning-fox.cc -o cunning-fox.js
multithread.js: multithread.cc
	/media/big/code-external/emsdk_portable/emscripten/master/em++ -std=c++11 -O1 multithread.cc -o multithread.js
257.js: 257.cc
	/media/big/code-external/emsdk_portable/emscripten/master/em++ -std=c++11 -O1 257.cc -o 257.js
111.js: 111.cc
	/media/big/code-external/emsdk_portable/emscripten/master/em++ -std=c++11 -O3 111.cc -o 111.js
clean:
	rm *.js
