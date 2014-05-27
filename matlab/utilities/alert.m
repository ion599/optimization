% poke me
WarnWave = 0.5 * [sin(1:.6:400), sin(1:.7:400), sin(1:.4:400)];
Audio = audioplayer(repmat(WarnWave,1,10), 22050);
play(Audio);