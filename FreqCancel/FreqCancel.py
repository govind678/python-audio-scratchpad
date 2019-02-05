# -----------------------------------------------------------------------------
#!/usr/bin/python -u
# encoding: utf-8
# 
# FreqCancel.py
# 
# Note: Using Python 2.7
# 
# Feb 2, 2019
# -----------------------------------------------------------------------------


import sys
import argparse
import time
import numpy as np

import wave
import struct

from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy import signal
import cmath
import math

from matplotlib import pyplot



# -----------------------------------------------------------------------------
# Read wave file by block
# 	- Using Python wave module
#	- Return multi-channel numpy n-dimensional array as Float32 scaled to [-1:1]
# -----------------------------------------------------------------------------
def readWaveFileByBlock(waveFileHandle, numFramesToRead, byteDepth, numChannels):

	inByteData = waveFileHandle.readframes(numFramesToRead)
	bytesJustRead = len(inByteData)
	samplesJustRead = bytesJustRead / byteDepth

	if byteDepth == 1:
		dataFormatString = "<%iB" % (samplesJustRead)
		scale = (np.iinfo(np.uint8).max / 2) + 1
		offset = scale
	elif byteDepth == 2:
		dataFormatString = "<%ih" % (samplesJustRead)
		scale = np.iinfo(np.int16).max
		offset = 0
	else:
		sys.exit("Error: Unsupported wav file byte depth: {}".format(byteDepth))

	outData = np.reshape(np.asarray(struct.unpack(dataFormatString, inByteData), dtype=np.float32), (samplesJustRead / numChannels, numChannels))
	outData = (outData - offset) / scale
	return outData



# -----------------------------------------------------------------------------
# Write wave file by block
# 	- Using Python wave module
#	- Accepts numpy Float32 array [-1:1]
# -----------------------------------------------------------------------------
def writeWaveFileByBlock(waveFileHandle, outFrames, byteDepth, numChannels):

	numFramesToWrite = outFrames.shape[0]

	if byteDepth == 1:
		dataFormatString = "<%iB" % (numFramesToWrite * numChannels)
		clip_min = 0.0
		clip_max = float(np.iinfo(np.uint8).max)
		scale = (clip_max / 2.0) + 1.0
		offset = scale
	elif byteDepth == 2:
		dataFormatString = "<%ih" % (numFramesToWrite * numChannels)
		clip_min = float(np.iinfo(np.int16).min)
		clip_max = float(np.iinfo(np.int16).max)
		scale = clip_max
		offset = 0 
	# elif byteDepth == 3:
		# dataFormatString = ""
	else:
		sys.exit("Error: Currently only supports 8 and 16 bit wave files. Unsupported {} bit(s)".format(byteDepth*8))

	outFrames = np.clip((outFrames + offset) * scale, clip_min, clip_max)
	outByteData = struct.pack(dataFormatString, *(np.ravel(outFrames)))
	waveFileHandle.writeframes(outByteData)



# -----------------------------------------------------------------------------
# checkWaveFileSanity: 
# 	- Ensure input file is within spec of this script
# -----------------------------------------------------------------------------
def checkWaveFileSanity(inWaveFile):
	numChannels = inWaveFile.getnchannels()
	byteDepth = inWaveFile.getsampwidth()
	sampleRate = inWaveFile.getframerate()
	if numChannels != 2:
		sys.exit("Error! This script only works on 2 channels of audio. Passed audio file with ({}) number of channels.".format(numChannels))
	if (byteDepth != 1) and (byteDepth != 2):
		sys.exit("Error: Currently only supports 8 and 16 bit wave files. Unsupported {} bit(s)".format(byteDepth*8))
	if sampleRate < 1000:
		sys.exit("Error: Incorrect sample rate: {}".format(sampleRate))



# -----------------------------------------------------------------------------
# dB2Lin: 
# 	- Convert decibel dB scale to linear
# -----------------------------------------------------------------------------
def dB2Lin(y):
	return (10.0 ** (y / 20.0))



# -----------------------------------------------------------------------------
# lin2dB: 
# 	- Convert linear to decibel dB scale
# -----------------------------------------------------------------------------
def lin2dB(y):
	return (20.0 * math.log(y, 10.0))



# -----------------------------------------------------------------------------
# RingBuffer class
# -----------------------------------------------------------------------------
class RingBuffer(object):
	"""A ring buffer for multi-dimensional float32 numpy arrays. Manages index wrapping around the first dimension."""
	def __init__(self, bufferShape):
		super(RingBuffer, self).__init__()
		self.__bufferLength = bufferShape[0]
		self.__readIdx = 0
		self.__writeIdx = 0
		self.__buffer = np.zeros(bufferShape, dtype=np.float32)

	def offsetReadIdx(self, offset):
		self.__readIdx = (self.__readIdx + offset) % self.__bufferLength

	def offsetWriteIdx(self, offset):
		self.__writeIdx = (self.__writeIdx + offset) % self.__bufferLength

	def read(self, length):
		indices = np.arange(self.__readIdx, self.__readIdx + length, 1, dtype=np.int16) % self.__bufferLength
		return self.__buffer[indices, :]

	def write(self, frames):
		indices = np.arange(self.__writeIdx, self.__writeIdx + frames.shape[0], 1, dtype=np.int16) % self.__bufferLength
		self.__buffer[indices, :] = frames

	def readAtIdx(self, startIdx, length):
		indices = np.arange(startIdx, startIdx + length, 1, dtype=np.int16) % self.__bufferLength
		return self.__buffer[indices, :]

	def writeAtIdx(self, frames, startIdx, length):
		indices = np.arange(startIdx, startIdx + length, 1, dtype=np.int16) % self.__bufferLength
		self.__buffer[indices, :] = frames

	def getReadIdx(self):
		return self.__readIdx

	def getWriteIdx(self):
		return self.__writeIdx





# -----------------------------------------------------------------------------------------------
# Main 
# -----------------------------------------------------------------------------------------------


# Start time
startTime = time.time()


# Parse command line args
parser = argparse.ArgumentParser(description='Input a 2 channel wave audio file. Perform FFT analysis and cancel elements of channel 2 from channel 1 and vice-versa.\
 												Then perform IFFT resynthesis and write output to wave audio file.')
parser.add_argument('filepath', metavar='filepath', nargs=1,
                    help='relative filepath of audio file (ending in .wav)')
args = parser.parse_args()


# Read audio file path from command line args
inFilepath = args.filepath[0]
if inFilepath[-4:] != '.wav':
	sys.exit("Incorrect filename ({}) . Must end in '.wav'".format(inFilepath))


# Pre-gain parameter (in dB, 0 -> no gain)
preGainDB = 0


# Open input wave file
try:
	inWaveFile = wave.open(inFilepath, 'rb')
except Exception as e:
	sys.exit("Error opening input audio file ({})".format(e))


# Ensure input wav file is within spec
checkWaveFileSanity(inWaveFile)


# Gather input wave file parameters
numFrames = inWaveFile.getnframes()
numChannels = inWaveFile.getnchannels()
byteDepth = inWaveFile.getsampwidth()
sampleRate = inWaveFile.getframerate()
inWaveFileParams = inWaveFile.getparams()


# Create and open output wave file
outFilepath = inFilepath[0:-4] + "_out_freqCancel" + ".wav"
try:
	outWaveFile = wave.open(outFilepath, 'wb')
except Exception as e:
	sys.exit("Error opening / creating output audio file ({})".format(e))


# Set output wave file format
outWaveFile.setparams(inWaveFileParams)	


# Audio processing block size (in frames), for FFT / IFFT
blockSize = 2048


# Overlap amount
overlapAmount = 0.75
hopSize = int((1.0 - overlapAmount) * blockSize)


# Setup ring buffers
inRingBuffer = RingBuffer((blockSize * 2, numChannels))
outRingBuffer = RingBuffer((blockSize * 2, numChannels))


# Window function - same for both analysis and synthesis
window = signal.hann(blockSize, 0)
windowAmpCorrection = np.size(window) / np.sum(window)


# Polar representation of Discrete Fourier Transform / DFT (magnitude and phase)
pol1 = np.zeros((blockSize, 2))
pol2 = np.zeros((blockSize, 2))


# Rect / Cartesian representation of DFT (Real and Imaginary) 
dftOut1 = np.zeros(blockSize, dtype=np.complex_)
dftOut2 = np.zeros(blockSize, dtype=np.complex_)



# -------------------------------------------------------------
# Start block-wise audio processing
# -------------------------------------------------------------
for frame in xrange(0, numFrames, hopSize):

	# Read wave audio block as Float32 n-dim numpy array
	inHop = readWaveFileByBlock(inWaveFile, hopSize, byteDepth, numChannels)
	numFramesRead = inHop.shape[0]

	# Apply pre-gain
	inHop = inHop * dB2Lin(preGainDB)

	# Write the 'hop' into input ring buffer
	inRingBuffer.writeAtIdx(inHop, frame, numFramesRead)

	# Read buffer size from input ring buffer
	sig = inRingBuffer.readAtIdx(frame + blockSize, blockSize)

	# Apply analysis window
	sig1 = np.multiply(sig[:,0], window)
	sig2 = np.multiply(sig[:,1], window)

	# Direct FFT
	dft1 = fft(sig1)
	dft2 = fft(sig2)

	# Convert cartesian (real + imag) to polar (magnitude, phase)
	for f in xrange(0, blockSize):
		pol1[f] = cmath.polar(dft1[f])
		pol2[f] = cmath.polar(dft2[f])

	# Get magnitude spectra
	#  - Fourier transform in polar form consists of the magnitude and phase spectra
	#  - Operations / processing (boosting, attenuating) are performed on the magnitude spectrum
	#  - Phase information needs to be preserved to perform Inverse FFT
	outMag1 = pol1[:,0]
	outMag2 = pol2[:,0]

	# Analysis window amplitude correction
	outMag1 = outMag1 * windowAmpCorrection
	outMag2 = outMag2 * windowAmpCorrection

	# Scale magnitude spectrum to [0.0 : 1.0]
	outMag1 = outMag1 / ((blockSize / 2.0) + 1.0)
	outMag2 = outMag2 / ((blockSize / 2.0) + 1.0)

	# Plot 'before' magnitude spectra
	# figure = pyplot.figure()
	# figure.suptitle("Before Magnitude Spectrum. Samples ({} : {}) / {}".format(frame, frame+blockSize, numFrames))
	# pyplot.plot(outMag1)
	# pyplot.plot(outMag2)
	# pyplot.show()

	# ----------------------------------------------------------------
	#! Interesting Stuff Here. 

	#  - The actual algorithm: iterating through the frequency bins,
	#  - compare to see if elements of spectrum 2 exist in spectrum 1,
	#  - then zero that particular frequency component (and vice-versa).
	#  - Improvements needed :)

	for f in xrange(0, blockSize):
		if outMag1[f] > outMag2[f]:
			outMag2[f] = 0.0
		elif outMag2[f] > outMag1[f]:
			outMag1[f] = 0.0


	#! End Interesting Stuff
	# ----------------------------------------------------------------

	# Plot 'after' magnitude spectra
	# figure = pyplot.figure()
	# figure.suptitle("After Magnitude Spectrum. Samples {} : {} / {}.".format(n, n+blockSize, numFrames))
	# pyplot.plot(outMag1)
	# pyplot.plot(outMag2)
	# pyplot.show()

	# Scale magnitude spectrum back
	outMag1 = outMag1 * ((blockSize / 2.0) + 1.0)
	outMag2 = outMag2 * ((blockSize / 2.0) + 1.0)

	# Convert polar back to cartesian
	for f in xrange(0, blockSize):
		dftOut1[f] = cmath.rect(outMag1[f], pol1[f,1])
		dftOut2[f] = cmath.rect(outMag2[f], pol2[f,1])

	# Inverse FFT
	iftOut1 = ifft(dftOut1)
	iftOut2 = ifft(dftOut2)

	# Get real
	out1 = np.real(iftOut1)
	out2 = np.real(iftOut2)

	# ----------------------------------------------------------
	#! Applying an output / synthesis window 
	#! turns out to be causing more issues than solving them.
	#! Skip for now.

	# Apply output window
	# out1 = np.multiply(out1, window)
	# out2 = np.multiply(out2, window)

	# Synthesis window amplitude correction
	# out1 = out1 / (windowAmpCorrection * 2.0)
	# out2 = out2 / (windowAmpCorrection * 2.0)
	# ----------------------------------------------------------

	# Read from output ring buffer
	out = outRingBuffer.readAtIdx(frame, blockSize)

	# Overlap-add with amplitude correction
	out[:,0] = out[:,0] + (out1 * (1.0 - overlapAmount))
	out[:,1] = out[:,1] + (out2 * (1.0 - overlapAmount))

	# Write into output ring buffer
	outRingBuffer.writeAtIdx(out, frame, blockSize)

	# Read 'hop' that was just finished from output ring buffer and write to WAV file
	writeWaveFileByBlock(outWaveFile, outRingBuffer.readAtIdx(frame, numFramesRead), byteDepth, numChannels) 

	# Clear output 'hop' for next overlap-add
	outRingBuffer.writeAtIdx(np.zeros((numFramesRead, numChannels)), frame, numFramesRead)


# --- End audio block processing --- #	


# Close files
inWaveFile.close()
outWaveFile.close()


# Finish
print "Finished! Elapsed time: {}. Output at : {}".format((time.time() - startTime), outFilepath)
