```python
from opentrons import protocol_api

def run(protocol: protocol_api.ProtocolContext):
    tiprack = protocol.load_labware(load_name = 'opentrons_96_filtertiprack_200ul', location = 2)
    plate = protocol.load_labware(load_name = 'corning_12_wellplate_6.9ml_flat', location = 3)
    pipette = protocol.load_instrument(instrument_name = 'p1000_single_gen2', mount='left')
    tiprack = protocol.load_labware('opentrons_96_filtertiprack_200ul', '2')
    pipette.pick_up_tip(tiprack.wells()[0])
```