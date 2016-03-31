#!/usr/bin/python

import argparse
import os
import sys
import subprocess
import inspect

file_suffix = ".js"
v8_options = ["--allow-natives-syntax", "--expose-gc",]
llv8_options = v8_options + [
    "--llvm-filter=foo*",
    "--noturbo",
    "--noturbo-asm",
#    "--noconcurrent-recompilation",
#    "--noconcurrent-osr",
#    "--noinline-new",
#    "--nouse-osr",
#    "--nouse-inlining",
    ]

null_file = open("/dev/null", "w")

arg_parser = argparse.ArgumentParser(
    description="Run v8_path on tests and compare llv8 outputs with pure v8."
    "Tests are expected to have " + file_suffix + " suffix and contain "
    "a function foo() which will be llvmed (other funcitons won't be)." )
arg_parser.add_argument('--filter',
                        help="Use only tests which have FILTER as a substring")
arg_parser.add_argument('--exclude',
                        help="The set of tests to be skipped (whose filename contains the substring)")
arg_parser.add_argument('--src_root',
                        default=os.path.dirname(os.path.realpath(__file__)),
                        help="Root directory with tests")
arg_parser.add_argument('v8_path',
                        nargs='?', # 0 or 1
                        default="/home/vlad/code/blessed-v8/v8/out/x64.debug/d8")
args = arg_parser.parse_args()

print args
v8_path = args.v8_path
src_root = args.src_root

class WrongAnswerException(Exception):
    pass

def do_test(filename):
    llv8_out = subprocess.check_output([v8_path] + llv8_options + [filename], stderr=null_file)
    v8_out = subprocess.check_output([v8_path] + v8_options + [filename], stderr=null_file)
    split_lambda = lambda output: filter(lambda x: x, output.split("\n"))
    llv8_out = split_lambda(llv8_out)
    v8_out = split_lambda(v8_out)
    if len(v8_out) == 0 and len(llv8_out) == 0:
        return
    elif len(v8_out) == 0 or len(llv8_out) == 0:
        raise WrongAnswerException("llv8 error: WA | empty output")
    elif llv8_out[-1] != v8_out[-1]:
        print "llv8:\t", llv8_out[-1]
        print "v8:\t", v8_out[-1]
        raise WrongAnswerException("llv8 error: WA")

failed = []
tested_cnt = 0
for root, dirs, files in os.walk(src_root):
    lst = [root + '/' + i for i in files if i.endswith(file_suffix)]
    for src_file in lst:
        if args.exclude and args.exclude in src_file: continue
        if args.filter and args.filter not in src_file: continue
        tested_cnt += 1
        try:
            print src_file
            do_test(src_file)
            print "\tOK"
        except Exception as e:
            failed += [src_file]
            print "\tFAILED!"
            print e
print "\n=======RESULTS======="
print str(len(failed)) + "/" + str(tested_cnt), "tests failed"
for test in failed:
    print test
null_file.close()
