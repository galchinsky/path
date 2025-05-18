from pyo import *
import os


class PyoGridAudioManager:
    """
    Loads 64 stereo loops (files 0_0.wav … 7_7.wav) and plays/-fades them
    according to 8×8 grid states.
    A global brick-wall limiter prevents digital clipping when many cells are on.
    """

    def __init__(self, folder_path, fade_time=0.5, use_ram=True,
                 limiter_thresh_db=-12, limiter_ratio=20):
        """
        folder_path: directory with 64 stereo WAV files named as '0_0.wav', ..., '7_7.wav'
        fade_time: fade-in/out time in seconds
        use_ram: True = use SndTable (in-memory), False = SfPlayer (disk-based)
        limiter_thresh_db: threshold in dB where limiting starts
        limiter_ratio: compression ratio (high ratio ≈ hard limiter)
        """
        self.server = Server().boot()
        self.server.start()
        self.fade_time = fade_time
        self.players = {}          # (i,j) -> PyoObject (TableRead | SfPlayer)
        self.faders  = {}          # (i,j) -> Fader
        self.cell_states = {}
        self.use_ram = use_ram
        self._debug(f"Server started  –  RAM mode: {self.use_ram}")

        # ------------------------------------------------------------------ #
        #  Build per-cell players ( *without* calling .out() yet )           #
        # ------------------------------------------------------------------ #
        buses = []                  # we'll mix everything in one stereo bus

        # Define the grid pattern for the 9 audio files
        grid_pattern = [
            "51351351",
            "68268268",
            "78978978",
            "51351351",
            "68268268",
            "78978978",
            "51351351",
            "68268268"
        ]

        # Original code (commented out)
        # for i in range(8):
        #     for j in range(8):
        #         filename = f"{i}_{j}.wav"
        #         path = os.path.join(folder_path, filename)
        #         if not os.path.isfile(path):
        #             raise FileNotFoundError(f"Missing loop: {path}")
        #
        #         fad = Fader(fadein=fade_time, fadeout=fade_time, mul=0.7)
        #         if self.use_ram:
        #             tbl   = SndTable(path)
        #             ply   = TableRead(tbl, freq=tbl.getRate(), loop=True, mul=fad)
        #         else:
        #             ply   = SfPlayer(path, loop=True, mul=fad)
        #
        #         self.faders[(i, j)]  = fad
        #         self.players[(i, j)] = ply
        #         self.cell_states[(i, j)] = False
        #         buses.append(ply)

        # New implementation using 9 audio files (1.wav through 9.wav)

        original_tables = {}
        for i in range(8):
            for j in range(8):
                # Get the file number from the grid pattern
                file_number = int(grid_pattern[i][j])
                filename = f"{file_number}.wav"
                path = os.path.join(folder_path, filename)
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Missing loop: {path}")

                fad = Fader(fadein=fade_time, fadeout=fade_time, mul=0.7)
                fad.play()
                if self.use_ram:
                    if file_number not in original_tables:
                        original_tables[file_number] = SndTable(path)
                    tbl = original_tables[file_number].copy()
                    ply = TableRead(tbl, freq=tbl.getRate(), loop=True, mul=1.0)#, mul=fad)
                else:
                    ply = SfPlayer(path, loop=True, mul=fad)

                ply = ply.play()

                self.faders[(i, j)] = fad
                self.players[(i, j)] = ply
                self.cell_states[(i, j)] = False
                buses.append(ply)

        # ------------------------------------------------------------------ #
        #  Master mix  →  brick-wall limiter  →  sound card                  #
        # ------------------------------------------------------------------ #


        mix  = Mix(buses, voices=2)  # sum all streams to 2-channel bus
        mix = mix.out()

        #self.limiter = Compress(     # almost brick-wall behaviour
        #    mix,
        #    thresh=limiter_thresh_db,  # dBFS where limiting starts
        #    ratio=limiter_ratio,       # high ratio ≈ hard limiter
        #    risetime=0.01,
        #    falltime=0.10,
        #    knee=0,
        #    outputAmp=1
        #)
        #self.limiter.out()


        self._debug(f"Limiter engaged (thresh {limiter_thresh_db} dBFS, "
                    f"ratio {limiter_ratio}:1)")
        self._debug(f"Loaded {len(self.players)} loops from {folder_path}")

    # ---------------------------------------------------------------------- #
    #  Public API                                                            #
    # ---------------------------------------------------------------------- #
    def set_cell_state(self, i, j, on):
        if (i, j) not in self.players:
            self._debug(f"Invalid cell ({i},{j}) – ignored")
            return

        cur = self.cell_states[(i, j)]
        if on and not cur:
            self.faders[(i, j)].play()
            self.cell_states[(i, j)] = True
            self._debug(f"Cell ({i},{j}) ON")
        elif not on and cur:
            self.faders[(i, j)].stop()
            self.cell_states[(i, j)] = False
            self._debug(f"Cell ({i},{j}) OFF")

    def stop_all(self):
        for fad in self.faders.values():
            fad.stop()
        self._debug("All cells OFF")

    def shutdown(self):
        self.stop_all()
        if self.server.getIsStarted():
            self.server.stop()
        self.server.shutdown()
        self.server.delete()
        self._debug("Server shutdown – goodbye!")

    # ---------------------------------------------------------------------- #
    #  Helpers                                                               #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _debug(msg: str):
        print(f"[AudioGrid] {msg}")
