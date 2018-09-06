#!/usr/bin/python3

import os
import sys, getopt
import random

def main(argv):
    print("Starting...")

    inSourcePath = argv[1]
    inTargetPath = argv[2]
    outDir = argv[3]
    inSourceHandle = open(inSourcePath, "r")
    inTargetHandle = open(inTargetPath, "r")

    docId = 0
    totLines = 0
    cont = True

    while cont:
        numLines = random.randint(11,300)
        totLines += numLines
        print(totLines)
        dir = outDir + "/" + str(docId)
        os.mkdir(dir)

        outSourceHandle = open(dir + "/fr.txt", "w")
        outTargetHandle = open(dir + "/en.txt", "w")

        for lineNum in range(numLines):
            sourceLine = inSourceHandle.readline()

            if sourceLine == '':
                cont = False
                break

            targetLine = inTargetHandle.readline()

            #print(sourceLine)
            #print(targetLine)

            outSourceHandle.write(sourceLine)
            outTargetHandle.write(targetLine)

        docId += 1

        outSourceHandle.close()
        outTargetHandle.close()


    inSourceHandle.close()
    inTargetHandle.close()

    print("Finished")

if __name__ == "__main__":
    main(sys.argv)
