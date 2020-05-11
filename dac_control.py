# DAC control
# Author: Rajkumar Kubendran
#

VDAC_NUM = 2
IDAC_NUM = 1
DAC_NUM_CHANNEL = 8

VDAC_VOLTAGES = [
	[0.2, 0.5, 0.25, 0.2, 0.0, 0.0, 0.0, 0.5],
    [0.1, 1.0, 0.365, 0.6, 0.2, 1.0, 0.6, 0.1],
]

IDAC_VOLTAGES = [
	[0.6, 0.47, 0.6, 0.45, 0.0, 0.0, 0.0, 0.0], #0.47, 0.45
]

VDAC_VREFS = [1.2, 1.2]
IDAC_VREFS = [1.2]


def idac_program_single(dev, chnl_idx, voltage, vref, no_op=False):
	if no_op:
		cmd = dac_translate_cmd(chnl_idx, voltage, vref, cmd=0b1111)
	else:
		cmd = dac_translate_cmd(chnl_idx, voltage, vref)
	dev.UpdateWireOuts()
	status = dev.GetWireOutValue(0x23)
	if status & 0b01 != 0: # Check whether the FIFO is full
		print('Current DAC FIFO is full.')
		return
	dev.SetWireInValue(0x10, cmd & 0x0000ffff)	# command
	dev.SetWireInValue(0x11, ((cmd & 0xffff0000) >> 16))  # command
	dev.UpdateWireIns()
	dev.ActivateTriggerIn(0x40, 0)


def idac_program_single_daisy(dev, dac_idx, chnl_idx, voltage, vrefs=IDAC_VREFS):
	for i in range(IDAC_NUM-1, -1, -1):
		if i == dac_idx:
			idac_program_single(dev, chnl_idx, voltage, vrefs[i])
		else:
			idac_program_single(dev, chnl_idx, voltage, vrefs[i], no_op=True)
			while True:
				dev.UpdateWireOuts()
				status = dev.GetWireOutValue(0x23)
				if status & 0b10 != 0:
					break
	dev.ActivateTriggerIn(0x41, 0)
	print('Updated Current DAC %d channel %d voltage to %fV.' % (dac_idx, chnl_idx, voltage))

def vdac_program_single(dev, chnl_idx, voltage, vref, no_op=False):
	if no_op:
		cmd = dac_translate_cmd(chnl_idx, voltage, vref, cmd=0b1111)
	else:
		cmd = dac_translate_cmd(chnl_idx, voltage, vref)
	dev.UpdateWireOuts()
	status = dev.GetWireOutValue(0x23)
	if status & 0b0100 != 0: # Check whether the FIFO is full
		print('Voltage DAC FIFO is full.')
		return
	dev.SetWireInValue(0x10, cmd & 0x0000ffff)	# command
	dev.SetWireInValue(0x11, ((cmd & 0xffff0000) >> 16))  # command
	dev.UpdateWireIns()
	dev.ActivateTriggerIn(0x40, 1)


def vdac_program_single_daisy(dev, dac_idx, chnl_idx, voltage, vrefs=VDAC_VREFS):
	for i in range(VDAC_NUM-1, -1, -1):
		if i == dac_idx:
			vdac_program_single(dev, chnl_idx, voltage, vrefs[i])
		else:
			vdac_program_single(dev, chnl_idx, voltage, vrefs[i], no_op=True)
			while True:
				dev.UpdateWireOuts()
				status = dev.GetWireOutValue(0x23)
				if status & 0b1000 != 0:
					break
	dev.ActivateTriggerIn(0x41, 1)
	print('Updated Voltage DAC %d channel %d voltage to %fV.' % (dac_idx, chnl_idx, voltage))

def dac_program_all(dev, voltages=IDAC_VOLTAGES, vrefs=IDAC_VREFS):
	voltages = IDAC_VOLTAGES
	vrefs = IDAC_VREFS
	for c in range(DAC_NUM_CHANNEL):
		for i in range(IDAC_NUM):
			volt = voltages[IDAC_NUM-1-i][c]
			idac_program_single(dev, c, volt, vrefs[IDAC_NUM-1-i])
			while True:
				dev.UpdateWireOuts()
				status = dev.GetWireOutValue(0x23)
				if status & 0b10 != 0:
					print('Programed Current DAC %d Channel %d to %fV.' % (IDAC_NUM-i, c, volt))
					break
		dev.ActivateTriggerIn(0x41, 0)
		# print('Updated channel %d output' % c)

	voltages = VDAC_VOLTAGES
	vrefs = VDAC_VREFS
	for c in range(DAC_NUM_CHANNEL):
		for i in range(VDAC_NUM):
			volt = voltages[VDAC_NUM-1-i][c]
			vdac_program_single(dev, c, volt, vrefs[VDAC_NUM-1-i])
			while True:
				dev.UpdateWireOuts()
				status = dev.GetWireOutValue(0x23)
				if status & 0b1000 != 0:
					print('Programed Voltage DAC %d Channel %d to %fV.' % (VDAC_NUM-i, c, volt))
					break
		dev.ActivateTriggerIn(0x41, 1)
		# print('Updated channel %d output' % c)


def dac_translate_cmd(chnl_idx, voltage, vref, cmd=0b0011):
	if chnl_idx < 0 or chnl_idx > 7:
		print('DAC channel index is out of range.')
		cmd = 0b1111
	if voltage < 0 or voltage > vref:
		print('DAC voltage is out of range.')
		cmd = 0b1111
	data = int(voltage / vref * 2**16)
	return (cmd << 20) | (chnl_idx << 16) | data & 0x00ffffff
