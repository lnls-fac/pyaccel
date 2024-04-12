"""."""

class Backend:

    def __init__(self):
        self._NUM_COORDS = None
        self._DIMS = None

        self.language  = "unknown"
        self.Element       = None
        self.PASS_METHODS  = None
        self.I_SHIFT       = None
        self.float_MAX     = None
        self._COORD_VECTOR = None
        self._COORD_MATRIX = None

        self.Accelerator   = None
        self.ElementVector = None

        self.marker        = None
        self.bpm           = None
        self.drift         = None
        self.matrix        = None
        self.hcorrector    = None
        self.vcorrector    = None
        self.corrector     = None
        self.rbend         = None
        self.quadrupole    = None
        self.sextupole     = None
        self.rfcavity      = None
        self.kickmap       = None

    def PassMethod(self, index):
        pass

    def Int(self, value):
        pass

    def FloatVector(self, arr=None):
        pass

    def VChamberShape(self, value):
        pass

    def get_array(self, pointer):
        pass

    def get_matrix(self, pointer):
        pass

    def set_array_from_vector(self, array, size, values):
        pass

    def set_array_from_matrix(self, array, shape, values):
        pass

    def force_set(self, obj, field, value):
        pass

    def get_size(self, obj):
        pass

    def get_acc_length(self, acc):
        pass

    def bkd_isinstance(self, obj, type):
        pass

    def isequal(self, this, other):
        pass

    def get_kicktable(self, index):
        pass

    def insertElement(self, lattice, element, index):
        pass

    def set_polynom(self, polynom, val):
        pass
