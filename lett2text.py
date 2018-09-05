#!/usr/bin/env python3

import os
import sys, getopt
import base64
import subprocess

###################################################################
def systemCheck(cmd):
    sys.stderr.write("Executing:" + cmd + "\n")
    sys.stderr.flush()

    subprocess.check_call(cmd, shell=True)

###################################################################
def main(argv):
    print("Starting...")

    mosesScriptDir = "/home/hieu/workspace/github/mosesdecoder/scripts"

    nunFiles = 0
    for line in sys.stdin:
    #for line in fileIn:
        #sys.stdout.write(line)
        toks = line.split("\t")
        #print(len(toks))
        decoded = str(base64.b64decode(toks[5]).decode('UTF-8'))
        #print(decoded)

        lang = toks[0]

        fileW = open("/tmp/hh", mode="w", encoding="utf-8")
        #fileW = open("/home/hieu/david/extracted/" + str(nunFiles) + "." + lang, mode="w", encoding="utf-8")

        for lineDecoded in decoded.splitlines():
            lineDecoded = lineDecoded.strip()
            if len(lineDecoded) > 0: 
                fileW.write(lineDecoded + "\n")
                
        fileW.close()

        cmd = 'cat /tmp/hh | ' + mosesScriptDir + '/tokenizer/tokenizer.perl -l ' + lang \
              + ' | ' + mosesScriptDir + '/tokenizer/deescape-special-chars.perl ' \
              + ' | ' + mosesScriptDir + '/recaser/truecase.perl --model training/train_truecase-model.' + lang \
              + " > /home/hieu/david/extracted/" + str(nunFiles) + "." + lang
        systemCheck(cmd)
            
        # html
        #decoded = str(base64.b64decode(toks[4]).decode('UTF-8'))
        #print(decoded)

        #fileW = open("/tmp/" + str(i) + ".html", "w")
        #fileW.write(decoded)
        #fileW.close()

        nunFiles += 1

    # extract parallel
    cmd = 'python3 /home/hieu/workspace/github/paracrawl/parSentExtract.hieu/extract.py ' \
          + '--checkpoint_dir /home/hieu/david/tflogs ' \
          + '--extract_dir /home/hieu/david/extracted ' \
          + '--source_vocab_path /home/hieu/david/training/vocabulary.source ' \
          + '--target_vocab_path /home/hieu/david/training/vocabulary.target ' \
          + '--source_output_path t --target_output_path t2 --score_output_path t3 ' \
          + '--source_language en  --target_language fr --decision_threshold .99'
    systemCheck(cmd)


    print("Finished")

if __name__ == "__main__":
    main(sys.argv[1:])

