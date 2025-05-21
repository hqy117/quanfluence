 #------------------------------------------------------------------------------
#
#            (c) Copyright 2024 by QUANFLUENCE PRIVATE LIMITED
#                          All rights reserved.
#
#   Trade Secret of QUANFLUENCE PRIVATE LIMITED  Do not disclose.
#
#   Use of this file in any form or means is permitted only
#   with a valid, written license agreement with QUANFLUENCE PRIVATE LIMTED.
#   The licensee shall strictly limit use of information contained herein
#   to the conditions specified in the written license agreement.
#
#   Licensee shall keep all information contained herein confidential
#   and shall protect same in whole or in part from disclosure and
#   dissemination to all third parties.
#
#                         QUANFLUENCE PRIVATE LIMITED
#                        E-Mail: assist@quanfluence.com
#                             www.quanfluence.com
#
# -----------------------------------------------------------------------------

import sys
import dimod
from quanfluence_sdk import QuanfluenceClient

# ******* Signing in *************

client = QuanfluenceClient()
client.signin('rochester_user0','Rochesteruser@123')   # Preconfigured - DO NOT CHANGE

device_id = 16                              # Preconfigured - DO NOT CHANGE

# ******* Executing the uploaded QUBO on the server *******

device = client.get_device(device_id)
print(device)

result = client.execute_device_qubo_file(device_id, 'max_cut_triangle_21564ef7-b83b-4576-8722-469a7b78ef4c.qubo')     # ENTER THE FILENAME OBTAINED AFTER UPLOADING THE QUBO
print(result)
