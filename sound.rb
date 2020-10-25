# Welcome to Sonic Pi

sample "d:/Python/@morze/sample/seascape.wav", rate: 1, amp: 0.15

live_loop :flibble do
  sample :perc_bell, rate: 3, amp: 0.25
  sleep 4
end

live_loop :wegorze_main do
  with_fx :echo do
    with_fx :reverb do
      sleep rrand(4, 12)
      sample "d:/Python/@morze/sample/wegorze.wav", rate: [0.5, -0.5].choose, amp: rrand(0.05,0.1), pan: rrand(-1, 1)
    end
  end
end

live_loop :wegorze do
  with_fx :echo do
    with_fx :reverb do
      sleep rrand(8, 24)
      sample "d:/Python/@morze/sample/wegorze.wav", rate: [-1, 1].choose, amp: rrand(0.05,0.15), pan: rrand(-1, 1)
    end
  end
end

live_loop :dorsze do
  with_fx :echo do
    with_fx :reverb do
      sleep rrand(12, 64)
      sample "d:/Python/@morze/sample/dorsze.wav", rate: [-1, -1].choose, amp: rrand(0.1,0.1), pan: rrand(-1, 1)
    end
  end
end

live_loop :haunted do
  sample :perc_bell, rate: rrand(0.75,1.25), amp: rrand(0.01, 0.04), pan: rrand(-1, 1)
  sleep rrand(4, 8)
end

use_synth :piano
use_bpm 60
use_random_seed 3
use_debug false

with_fx :reverb do
  with_fx(:echo, delay: 0.5, decay: 8) do
    live_loop :echoes do
      play chord([:c2, :c3, :c4, :c5].choose, :minor).choose, cutoff: rrand(40, 100), amp: 0.1, attack: 0, release: rrand(1, 2), cutoff_max: 110
      sleep [0.5, 1, 1, 2, 2, 2].choose
    end
  end
end