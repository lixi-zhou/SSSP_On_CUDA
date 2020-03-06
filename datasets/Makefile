WGET=wget 
UNZIP=gzip -d

all: download unzip 

download:
	$(WGET) http://snap.stanford.edu/data/higgs-social_network.edgelist.gz
	$(WGET) http://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
	$(WGET) http://snap.stanford.edu/data/soc-pokec-relationships.txt.gz

unzip: 
	$(UNZIP) *.gz
