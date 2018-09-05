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

    mosesScriptDir = argv[1] # /home/hieu/workspace/github/mosesdecoder/scripts
    extractedDir = argv[2]
    
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
              + ' > ' + extractedDir + str(nunFiles) + "." + lang
        systemCheck(cmd)
            
        # html
        #decoded = str(base64.b64decode(toks[4]).decode('UTF-8'))
        #print(decoded)

        #fileW = open("/tmp/" + str(i) + ".html", "w")
        #fileW.write(decoded)
        #fileW.close()

        nunFiles += 1

    # extract parallel
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    print("scriptDir", scriptDir)
    
    cmd = 'python3 ' + scriptDir + '/extract.py ' \
          + ' --checkpoint_dir /home/hieu/david/tflogs ' \
          + ' --extract_dir ' + extractedDir \
          + ' --source_vocab_path /home/hieu/david/training/vocabulary.source ' \
          + ' --target_vocab_path /home/hieu/david/training/vocabulary.target ' \
          + ' --source_output_path t --target_output_path t2 --score_output_path t3 ' \
          + ' --source_language en  --target_language fr --decision_threshold .99'
    systemCheck(cmd)


    print("Finished")

if __name__ == "__main__":
    #scriptDir = os.path.dirname(os.path.realpath(__file__))
    #print("scriptDir", scriptDir)
    
    main(sys.argv)

