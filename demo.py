import argparse
import cv2
from screeninfo import get_monitors
import numpy as np
from tensorflow import lite as tflite
import pyaudio
from threading import Thread
import traceback as tb
import sys
from audio import soundscape
from itertools import cycle, count
from os import listdir
from os.path import isfile, join

from enum import Enum


AUDIO_SAMPLES = np.array([], dtype='float32')

BUFFER_SIZE = 1024
IMAGE_CHANNELS = 3
ACTIVE_COLORMAP = None
COLORMAPS = [cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_CIVIDIS, cv2.COLORMAP_BONE, cv2.COLORMAP_INFERNO, cv2.COLORMAP_JET, None]
SPACING = 0.0225
BORDER_COLOR = 128
TEXT_COLOR = (255, 255, 255)
NUMBER_OF_RESULTS = 15
MAXIMA = {}
STREAM = False
PAUSE = False
FRAME_NR = 0

OUTPUT_IDX = {'spec1': 220, 'spec2': 261, 'conv0': 266, 'block1': 294, 'block2': 370, 'block3': 465, 'block4': 522, 'post_conv': 544, 'pooling': 545, 'class': 546}
GRID_WIDTH = {'spec1': 1, 'spec2': 1, 'conv0': 2, 'block1': 2, 'block2': 2, 'block3': 2, 'block4': 3, 'post_conv': 6, 'pooling': 11, 'class': 30}
SCREEN_WIDTH = {'spec1': 0.2, 'spec2': 0.2, 'conv0': 0.2, 'block1': 0.125, 'block2': 0.1, 'block3': 0.1, 'block4': 0.1, 'post_conv': 0.1, 'pooling': 0.1, 'class': 0.2, 'bar_width': 0.05}

class RUNNING_MODES(Enum):
    PLAYBACK = 1
    LIVE = 2

def load(frame_width, frame_height, width_scaling, modelpath, labelpath):

    global interpreter, input_details, output_details, LABELS, width, height, SCREEN_WIDTH

    # Calculate the sum of the current values
    total = sum(SCREEN_WIDTH.values())

    # Normalize the values
    SCREEN_WIDTH = {key: (value / total) * width_scaling for key, value in SCREEN_WIDTH.items()}

    # Load labels file
    LABELS = []
    with open(labelpath, 'r') as f:
        for line in f:
            label = line.strip().split('_')[1]
            label = label.replace('Ã¤', 'ä').replace('Ã¶', 'ö').replace('Ã¼', 'ü').replace('ÃŸ', 'ß')
            label = label.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')

            label2 = line.strip().split('_')[0]
            label2 = label2.replace(' ', '_')
            LABELS.append([label2,label])

    # Load model
    interpreter = tflite.Interpreter(model_path=modelpath, experimental_preserve_all_tensors=True, num_threads=4)

    # Allocate tensors
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create window
    cv2.namedWindow('demo', cv2.WINDOW_NORMAL)

    # Get screen resolution
    screen = get_monitors()[0]
    screen_width = screen.width
    screen_height = screen.height

    primary_screen = get_monitors()[0]
    primary_screen_width = screen.width
    primary_screen_height = screen.height

    if frame_width == -1 and frame_height == -1:
        width = screen_width
        height = screen_height

        # Show image in window full screen
        cv2.setWindowProperty('demo', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty('demo', height, width)

    else:
        # Ensure the window size does not exceed the screen size
        width = min(frame_width, screen_width)
        height = min(frame_height, screen_height)

        # Set window position
        cv2.moveWindow('demo', 0, 0)

        # Set window size
        cv2.resizeWindow('demo', width, height)

    return (width, height)

def record():

    global AUDIO_SAMPLES

    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
        
    input_device_idx = -1
    for i in range(numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        device_name = device_info['name']
        print(f"Device {i}: {device_name}")

    try:
        if input_device_idx == -1:
            input_device_idx = int(input("Wähle ein Gerät für die Audio-Eingabe: \n"))
    except KeyboardInterrupt:
            print("Exiting...")
            sys.exit()

    # Print the default device information
    print("Device Info:")
    device_info = p.get_device_info_by_index(input_device_idx)
    for key, value in device_info.items():
        print(f"{key}: {value}")

    # Open microphone stream
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, 
                    channels=1, 
                    rate=48000, 
                    input=True, 
                    input_device_index=input_device_idx,
                    frames_per_buffer=BUFFER_SIZE)

    # Record audio
    while STREAM:
        if not PAUSE:
            data = stream.read(BUFFER_SIZE, exception_on_overflow=False)
            data = np.frombuffer(data, 'float32')
            AUDIO_SAMPLES = np.concatenate((AUDIO_SAMPLES, data))
            AUDIO_SAMPLES = AUDIO_SAMPLES[-144000:]

    # Close microphone stream
    stream.stop_stream()

# DEBUG: For each layer, show output details
"""
def listTensors(interpreter):

    # Create dummy input
    dummy_input = np.zeros((1, 144000), dtype=np.float32)

    # Run model
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

    # Show output shape for all tensors
    for i in range(len(interpreter.get_tensor_details())):
        data = interpreter.get_tensor(i)
        print(i, interpreter.get_tensor_details()[i]["name"], data.shape)        

listTensors(interpreter)
sys.exit()
"""

def plotWaveform(sig, height, width):

    # Create image
    img = np.zeros((height, width), np.uint8)

    # No negative values
    sig = np.abs(sig)

    # Resize signal to width
    sig = cv2.resize(sig, (width, 1), interpolation=cv2.INTER_AREA)

    # Avoid NaNs
    sig = np.nan_to_num(sig)

    # Normalize signal
    if not 'wf' in MAXIMA:
        MAXIMA['wf'] = [np.max(sig)]
    else:
        MAXIMA['wf'].append(np.max(sig))
        MAXIMA['wf'] = MAXIMA['wf'][-100:]
    #sig = (sig - np.min(sig)) / ((np.max(sig) - np.min(sig)) + 0.000001)    
    sig = (sig - np.min(sig)) / ((np.max(MAXIMA['wf']) - np.min(sig)) + 0.000001)
    sig = sig[0] 

    # Plot signal
    for i in range(sig.shape[0]):
        img[int((1 - sig[i]) / 2 * img.shape[0]):int((1 - sig[i]) / 2 * img.shape[0] + sig[i] * img.shape[0]), i] = 255

    return img

def parseOutput(output_data, grid_width, frame_width, frame_height, name='', border_width=1, show_cell_border=True, border_color=128, frame_border_color=128, normalize=True, apply_relu=False, apply_sigmoid=False, min_value=0, threshold=0):

    # Determine grid height
    grid_height = int(np.ceil(output_data.shape[-1] / grid_width))

    # Apply relu
    if apply_relu:
        output_data = np.maximum(output_data, 0)

    # Apply sigmoid
    if apply_sigmoid:
        output_data = 1 / (1 + np.exp(-output_data))

    # Normalize output
    if normalize:
        if not name in MAXIMA:
            MAXIMA[name] = [np.max(output_data)]
        else:
            MAXIMA[name].append(np.max(output_data))
            MAXIMA[name] = MAXIMA[name][-25:]
        output_data = np.clip((output_data - np.min(output_data)) / (np.max(output_data) * 0.75 - np.min(output_data) + 0.000001), 0, 1) * 255
        #output_data = np.minimum(1, (output_data - np.min(output_data)) / (np.max(MAXIMA[name]) * 0.7 - np.min(output_data) + 0.00000001)) * 255
    else:
        output_data = output_data * 255

    # Apply threshold
    if threshold > 0:
        output_data[output_data < threshold] = 0

    # Set min value
    if min_value > 0:
        output_data[output_data < min_value] = min_value

    # Create dummy frame
    # Each grid cell is based on output shape
    # and has 1px white border
    cell_width = output_data.shape[2]
    cell_height = output_data.shape[1]
    frame = np.zeros((int(grid_height * cell_height), int(grid_width * cell_width), 1), np.uint8)

    # For each grid cell
    for i in range(output_data.shape[-1]):
        x = i % grid_width
        y = int(i / grid_width)

        # Add axxis to output
        output = np.expand_dims(output_data[0, :, :, i], axis=-1)

        # Put output in center of grid cell
        frame[y * cell_height:y * cell_height + output.shape[0], x * cell_width:x * cell_width + output.shape[1]] = output   

    # Resize frame to frame_width x frame_height and keep aspect ratio
    scale = frame_width / frame.shape[1]
    if frame.shape[0] * scale > frame_height:
        scale = frame_height / frame.shape[0]
    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)), interpolation=cv2.INTER_NEAREST )       
        

    # Add border between grid cells
    if border_width > 0:

        if show_cell_border:
            # Vertical borders
            for i in range(grid_width - 1):
                frame[:, int((i + 1) * cell_width * scale + border_width):int((i + 1) * cell_width * scale + border_width * 2)] = border_color

            # Horizontal borders
            for i in range(grid_height - 1):
                frame[int((i + 1) * cell_height * scale + border_width):int((i + 1) * cell_height * scale + border_width * 2), :] = border_color

        # Border around entire frame
        frame = cv2.copyMakeBorder(frame, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, value=frame_border_color)
    
    # Expand dims
    frame = np.expand_dims(frame, axis=-1)

    # Convert into RGB by repeating 3 times
    #if IMAGE_CHANNELS > 1:
    #    frame = np.repeat(frame, IMAGE_CHANNELS, axis=-1)

    # Convert into RGB by applying viridis colormap
    if IMAGE_CHANNELS > 1 and ACTIVE_COLORMAP is not None:
        frame = cv2.applyColorMap(frame, ACTIVE_COLORMAP)

    return frame

def add_text(frame, text, fontsize, posX, posY, rot = 270, offset_x_ = -1):
    headspace = 5
    if offset_x_ != -1:
        offset_x = 0
    else:
        offset_x = int((width * SPACING)/2)
    
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontsize, 1)
    text_frame = np.full((text_height+headspace, text_width+headspace, IMAGE_CHANNELS), 0, np.uint8)
    cv2.putText(text_frame, text, (0, 0+text_height), cv2.FONT_HERSHEY_SIMPLEX, fontsize, TEXT_COLOR, 1, cv2.LINE_AA)
    text_frame = np.rot90(text_frame, rot/90)
    add_block(frame, text_frame, posX - offset_x, posY)
    
def add_block(frame, block, posX, posY):
    frame[posY:posY + block.shape[0], posX:posX + block.shape[1]] = block
      
def get_output(layer):
    out_tensor = interpreter.get_tensor(OUTPUT_IDX[layer])
    out_block = parseOutput(out_tensor, GRID_WIDTH[layer], int(width * SCREEN_WIDTH[layer]), int(height * (1 - SPACING * 2)), name=layer, border_width=1, border_color=BORDER_COLOR, frame_border_color=BORDER_COLOR, apply_relu=True)
    return out_block, out_tensor.shape



def main(mixer, fontsize, width, height, af_cycle, clip_files, unicode_key_list, running_mode = RUNNING_MODES.PLAYBACK):

    global IMAGE_CHANNELS, TEXT_COLOR, ACTIVE_COLORMAP, PAUSE, FRAME_NR, STREAM, STREAM_WORKER
    
    active_clip = ""
    cm_cycle = cycle(COLORMAPS)

    # Start ambience loop immediately
    active_clip = next(af_cycle)
    mixer.add_clip("data/soundscapes/"+active_clip, when_seconds=0.0, gain_db=0, fade_in_ms=3000)

    # Loop until user press ESC
    while True:

        # Create black dummy frame at screen resolution
        frame = np.zeros((height, width, IMAGE_CHANNELS), np.uint8)
  
        if running_mode == RUNNING_MODES.PLAYBACK:
            # Read from stream    
            sig = mixer.get_current_samples()

            # Get a single 144k-sample window (interleaved = 1-D)
            #sig = mixer.get_current_samples(interleaved=True, max_samples=144000)
            sig = mixer.get_current_samples_144k().reshape(1, 144000)   # float32
            # Trim or left-pad with zeros so it's exactly 144k
            if sig.size >= 144000:
                sig = sig[-144000:]
            else:
                sig = np.pad(sig, (144000 - sig.size, 0), mode='constant')

            add_text(frame, "PLAYBACK MODE (File: {})".format(active_clip), fontsize, posX=int(width/2), posY=5, rot=0, offset_x_=0)
        elif running_mode == RUNNING_MODES.LIVE:
            # Read from stream        
            sig = AUDIO_SAMPLES.copy()

            # If signal is shorter than 144000 samples, pad with zeros
            if len(sig) < 144000:
                sig = np.pad(sig, (0, 144000 - len(sig)), 'constant')

            add_text(frame, "LIVE MODE", fontsize, posX=int(width/2), posY=5, rot=0, offset_x_=0)
        
        # Reshape signal to 1x144000
        sig = sig.reshape(1, 144000) 

        # Run model
        #t_start = time.time()
        interpreter.set_tensor(input_details[0]['index'], sig)
        interpreter.invoke()
        #print("Inference time: {} ms".format((time.time() - t_start) * 1000))

        #current_pos_x = int(width * SPACING) #startx
        #current_pos_y = int(height * SPACING) #starty

        default_spacing_x = int(width * SPACING)
        default_spacing_y = int(height * SPACING)

        # Get output for spectrogram layer
        out_tensor = interpreter.get_tensor(OUTPUT_IDX['spec2'])
        out_tensor = np.expand_dims(out_tensor, axis=-1)
        output_spec = parseOutput(out_tensor, GRID_WIDTH['spec2'], int(width * SCREEN_WIDTH['spec2']), int(height * (0.33 - SPACING * 2)), name='spec2', border_width=1, border_color=BORDER_COLOR, frame_border_color=BORDER_COLOR)
        output_spec_posX = default_spacing_x
        output_spec_posY = int(height * 0.15)
        add_block(frame, output_spec, output_spec_posX, output_spec_posY)
        add_text(frame, "Spectrogram, {}x{} pixel".format(out_tensor.shape[1], out_tensor.shape[2]), fontsize, output_spec_posX, posY=output_spec_posY-20, rot=0, offset_x_=0)

        # Get output for waveform layer
        waveform = plotWaveform(sig, out_tensor.shape[1], out_tensor.shape[2])   
        waveform = np.expand_dims(waveform, axis=0)  
        waveform = np.expand_dims(waveform, axis=-1)
        output_wave = parseOutput(waveform, 1, int(width * SCREEN_WIDTH['spec2']), int(height * (0.33 - SPACING * 2)), name='wave', border_width=1, border_color=BORDER_COLOR, frame_border_color=BORDER_COLOR)
        output_wave_posX = default_spacing_x
        output_wave_posY = default_spacing_y
        add_block(frame, output_wave, output_wave_posX, output_wave_posY)
        add_text(frame, "Audio input stream, 3s @ 48kHz", fontsize, output_wave_posX, posY=output_wave_posY-20, rot=0, offset_x_=0)
    
        # Get output for conv0 layer
        out_tensor = interpreter.get_tensor(OUTPUT_IDX['conv0'])
        output_conv0 = parseOutput(out_tensor, GRID_WIDTH['conv0'], int(width * SCREEN_WIDTH['conv0']), int(height * (0.725 - SPACING * 2)), name='conv0', border_width=1, border_color=BORDER_COLOR, frame_border_color=BORDER_COLOR, apply_relu=False)
        #output_conv0 = cv2.resize(output_conv0, (output_spec.shape[1], output_conv0.shape[0]), interpolation=cv2.INTER_NEAREST)
        output_conv0 = cv2.resize(output_conv0, (output_spec.shape[1], int(height - SPACING * 2 - output_spec_posY - output_spec.shape[0] * 2)), interpolation=cv2.INTER_NEAREST)
        if len(output_conv0.shape) == 2:
            output_conv0 = np.expand_dims(output_conv0, axis=-1)
        output_conv0_posX = default_spacing_x
        output_conv0_posY = height - default_spacing_y - output_conv0.shape[0] #from end of screen
        frame[output_conv0_posY:output_conv0_posY + output_conv0.shape[0], output_conv0_posX:output_conv0_posX + output_conv0.shape[1]] = output_conv0
        add_text(text="Pre-processing convolution"+", {} filters, {}x{} outputs".format(out_tensor.shape[-1], out_tensor.shape[1], out_tensor.shape[2]), fontsize=fontsize, frame=frame, posX=output_conv0_posX, posY=output_conv0_posY)

        # Get output for block 1 and add output plus text
        output_block1, tensor1_shape = get_output('block1')
        output_block1_posX = output_spec_posX + output_spec.shape[1] + default_spacing_x
        add_block(frame=frame, posX=output_block1_posX, posY=default_spacing_y, block=output_block1)
        add_text(text="Inverted ResBlock 1"+", {} filters, {}x{} outputs".format(tensor1_shape[-1], tensor1_shape[1], tensor1_shape[2]), fontsize=fontsize, frame=frame, posX=output_block1_posX, posY=default_spacing_y)

        # Get output for block 2 and add output plus text
        output_block2, tensor2_shape= get_output('block2')
        output_block2_posX = output_block1_posX + output_block1.shape[1] + default_spacing_x
        add_block(frame=frame, posX=output_block2_posX, posY=default_spacing_y, block=output_block2)
        add_text(text="Inverted ResBlock 2"+", {} filters, {}x{} outputs".format(tensor2_shape[-1], tensor2_shape[1], tensor2_shape[2]), fontsize=fontsize, frame=frame, posX=output_block2_posX, posY=default_spacing_y)

        # Get output for block 3 and add output plus text
        output_block3, tensor3_shape= get_output('block3')
        output_block3_posX = output_block2_posX + output_block2.shape[1] + default_spacing_x
        add_block(frame=frame, posX=output_block3_posX, posY=default_spacing_y, block=output_block3)
        add_text(text="Inverted ResBlock 3"+", {} filters, {}x{} outputs".format(tensor3_shape[-1], tensor3_shape[1], tensor3_shape[2]), fontsize=fontsize, frame=frame, posX=output_block3_posX, posY=default_spacing_y)
        
        # Get output for block 4 and add output plus text
        output_block4, tensor4_shape = get_output('block4')
        output_block4_posX = output_block3_posX + output_block3.shape[1] + default_spacing_x
        add_block(frame=frame, posX=output_block4_posX, posY=default_spacing_y, block=output_block4)
        add_text(text="Inverted ResBlock 4"+", {} filters, {}x{} outputs".format(tensor4_shape[-1], tensor4_shape[1], tensor4_shape[2]), fontsize=fontsize, frame=frame, posX=output_block4_posX, posY=default_spacing_y)
        
        # Get output for post conv and add output plus text
        output_post_conv, tensor_post_conv_shape = get_output('post_conv')
        output_post_conv_posX = output_block4_posX + output_block4.shape[1] + default_spacing_x
        add_block(frame=frame, posX=output_post_conv_posX, posY=default_spacing_y, block=output_block4)
        add_text(text="Post-pocessing convolution"+", {} filters, {}x{} outputs".format(tensor_post_conv_shape[-1], tensor_post_conv_shape[1], tensor_post_conv_shape[2]), fontsize=fontsize, frame=frame, posX=output_post_conv_posX, posY=default_spacing_y)

        # Get output for pooling and add output plus text
        out_tensor = interpreter.get_tensor(OUTPUT_IDX['pooling'])
        out_tensor = np.expand_dims(out_tensor, axis=0)
        out_tensor = np.expand_dims(out_tensor, axis=0)
        output_pooling = parseOutput(out_tensor, GRID_WIDTH['pooling'], int(width * SCREEN_WIDTH['pooling']), int(height * (1 - SPACING * 2)), name='pooling', border_width=1, border_color=BORDER_COLOR, frame_border_color=BORDER_COLOR, show_cell_border=True, apply_relu=False, threshold=48)
        output_pooling_posX = output_post_conv_posX + output_post_conv.shape[1] + default_spacing_x
        add_block(frame=frame, posX=output_pooling_posX, posY=default_spacing_y, block=output_pooling)
        add_text(text="Global average pooling"+", {} filters, {}x{} outputs".format(out_tensor.shape[-1], out_tensor.shape[1], out_tensor.shape[2]), fontsize=fontsize, frame=frame, posX=output_pooling_posX, posY=default_spacing_y)
            
        # Get class output and add output plus text
        out_tensor = interpreter.get_tensor(OUTPUT_IDX['class'])
        out_tensor = np.expand_dims(out_tensor, axis=0)
        out_tensor = np.expand_dims(out_tensor, axis=0)
        output_class = parseOutput(out_tensor, GRID_WIDTH['class'], int(width * SCREEN_WIDTH['class']), int(height * (1 - SPACING * 2)), name='class', border_width=1, border_color=0, frame_border_color=BORDER_COLOR, show_cell_border=True, normalize=False, apply_relu=False, apply_sigmoid=True, min_value=32)
        output_class_posX = output_pooling_posX + output_pooling.shape[1] + default_spacing_x
        add_block(frame=frame, posX=output_class_posX, posY=default_spacing_y, block=output_class)
        add_text(text="Class output, {} species".format(out_tensor.shape[-1]), fontsize=fontsize, frame=frame, posX=output_class_posX, posY=default_spacing_y)

        # Get N highest scoring classes with labels
        scores = interpreter.get_tensor(OUTPUT_IDX['class'])
        scores = 1 / (1 + np.exp(-scores))
        scores = scores[0]
        topN = np.argsort(scores)[::-1][:NUMBER_OF_RESULTS]
        topN_scores = scores[topN]
        topN_labels = [LABELS[i] for i in topN]
        
        # Show results
        for i in range(NUMBER_OF_RESULTS):
            bar_v_spacing = int(height * 0.025)
            bar_width = int(width * SCREEN_WIDTH['bar_width'])
            bar_height = int(height * (1 - SPACING * 1) / NUMBER_OF_RESULTS) - bar_v_spacing
            bar_posX = output_class_posX + output_class.shape[1] + default_spacing_x
            bar_posY = default_spacing_y + i * (bar_height + bar_v_spacing)

            # Draw bar background
            bc = (32, 32, 32) if topN_scores[i] > 0.3 else (32, 32, 32)
            cv2.rectangle(frame, (bar_posX, bar_posY), (bar_posX + bar_width, bar_posY + bar_height), bc, -1)   

            # Draw bar foreground based on score
            cv2.rectangle(frame, (bar_posX, bar_posY), (bar_posX + int(bar_width * topN_scores[i]), bar_posY + bar_height), TEXT_COLOR, -1)   

            # Draw label
            tc = max(32, min(255, topN_scores[i] * 2 * 255))
            tc = (int(tc), int(tc), int(tc))
            cv2.putText(frame, topN_labels[i][1], (bar_posX + bar_width + int(width * SPACING * 0.5), bar_posY + int(bar_height * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, fontsize, tc, 1, cv2.LINE_AA)   

        #if topN_scores[0] > 0.5:
        #    print ("top score")
        #    path = extractus.extract(topN_labels[0][0])
        #    im = cv2.imread(path)
        #    h,w,c = im.shape
        #    frame[0:h, 0:w] = im  #for top-left corner, 0:50 and 0:50 for my image; select your region here like 200:250

        # Add text on results
        add_text(text="Top {} results".format(NUMBER_OF_RESULTS), fontsize=fontsize, frame=frame, posX=bar_posX, posY=default_spacing_y)

        # Show image in window
        cv2.imshow('demo', frame)

        # DEBUG: Save every frame to file
        #cv2.imwrite('saved_frames/frame_{:04d}.png'.format(FRAME_NR), frame)        

        # Wait 1ms for user input
        key = cv2.waitKey(1)
        
        # If key is 'c', change channels
        if key == ord('c'):
            ACTIVE_COLORMAP = next(cm_cycle)

        # If key is 'p', pause
        elif key == ord('p'):
            PAUSE = not PAUSE
            cv2.waitKey(-1)
            PAUSE = not PAUSE

        # if key is 's', save image
        if key == ord('s'):
            cv2.imwrite('saved_frames/frame_{:04d}.png'.format(FRAME_NR), frame)

        # if key is 'a' switch to next audio file
        if key == ord('a'):
            
            active_clip = next(af_cycle)
            # Create black dummy frame at screen resolution
            frame = np.zeros((height, width, IMAGE_CHANNELS), np.uint8)
            add_text(text="LOADING SOUNDSCAPE {}...".format(active_clip), fontsize=fontsize, frame=frame, posX=int(width/3), posY=int(height/2), rot=0)
            cv2.imshow('demo', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # allow UI to refresh & catch a key
                break
            
            if(running_mode == RUNNING_MODES.LIVE):
                # Stop stream and wait for stream to finish
                STREAM = False
                STREAM_WORKER.join()

                # Initialize Soundscapemixer
                mixer = soundscape.SoundscapeMixer()
                mixer.start()
                running_mode = RUNNING_MODES.PLAYBACK

            mixer.clear_clips()
            mixer.add_clip("data/soundscapes/"+active_clip, when_seconds=0.0, gain_db=0, fade_in_ms=3000)
        
        # if key is 'r' switch to live mode
        if key == ord('r'):
            
            mixer.clear_clips()
            mixer.stop()

            # Start stream
            STREAM = True
            STREAM_WORKER = Thread(target=record, args=())
            STREAM_WORKER.start()

            running_mode = RUNNING_MODES.LIVE
        
         # if key is '1', toggle thunder
        
        for idx, i in enumerate(unicode_key_list):
            if key == i:
                if running_mode == RUNNING_MODES.LIVE:
                    pass
                else:
                    mixer.add_clip("data/clips/"+clip_files[idx], clip_type=soundscape.CLIP_TYPE.SOUND, when_seconds=0, gain_db=-4, fade_in_ms=50, fade_out_ms=500)

        # If user press ESC or q, break loop
        if key == 27 or key == ord('q'):
            print("Stopping..")
            if STREAM == True:
                STREAM = False
                STREAM_WORKER.join()
            else: 
                mixer.stop()
            break    

        # Increase frame number
        FRAME_NR += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="BirdNET-XRay Demo.")
    parser.add_argument('--resolution', type=str, default='fullscreen', help='Resolution of the window, e.g., "fullscreen" or "1024x768"')
    parser.add_argument('--scaling', type=float, default='1.0', help='Scaling factor for the width of the output elements. Default is 1.5, lower values might work better on smaller screens.')
    parser.add_argument('--fontsize', type=float, default='0.55', help='Font size for text elements. Default is 0.55.')
    parser.add_argument('--labelpath', type=str, default='model/BirdNET_GLOBAL_6K_V2.4_Labels_de.txt', help='Path to label file.')
    parser.add_argument('--modelpath', type=str, default='model/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite', help='Path to model file.')
    parser.add_argument('--audiopath', type=str, default='data/soundscapes', help='Path to soundscape files.')
    parser.add_argument('--soundpath', type=str, default='data/clips', help='Path to clips for soundscapes.')
    parser.add_argument('--live', type=bool, default=False, help='Whether to record live audio or use prerecorded audio files/soundscapes.')

    args = parser.parse_args()

    # Set resolution
    if args.resolution == 'fullscreen':
        frame_width = -1
        frame_height = -1
    else:
        try:
            frame_width, frame_height = map(int, args.resolution.split('x'))
        except ValueError:
            print("Invalid resolution format. Use 'fullscreen' or 'widthxheight'.")

    # Load data, model and window
    (width, height) = load(frame_width, frame_height, args.scaling, args.modelpath, args.labelpath)
    
    # Start recording or playback
    if(args.live):
        STREAM = True
        STREAM_WORKER = Thread(target=record, args=())
        STREAM_WORKER.start()
        running_mode = RUNNING_MODES.LIVE
    else:
        mixer = soundscape.SoundscapeMixer()
        mixer.start()
        audio_files = [f for f in listdir(args.audiopath) if join(args.audiopath, f).endswith('.mp3')]
        print('Available ambience files: ')
        for f in audio_files:
            print(f)
        clip_files = [f for f in listdir(args.soundpath) if join(args.soundpath, f).endswith('.mp3')]
        print('Available sound files for soundscape editing (toggle with 1-{}): '.format(min(9, len(clip_files))))
        unicode_key_list = []
        for idx, f in enumerate(clip_files):
            unicode_key_list.append(ord(str(idx)))
            print(f)

        af_cycle = cycle(audio_files)
        running_mode = RUNNING_MODES.PLAYBACK

    try:
        main(mixer, args.fontsize, width, height, af_cycle, clip_files, unicode_key_list, running_mode)
    except:
        tb.print_exc()

    # Destroy window
    cv2.destroyWindow('demo')

    # Stop recording
    STREAM = False

    mixer.stop()


