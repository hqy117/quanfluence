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

# ********** THIS FILE NEEDS TO BE RUN ONLY ONCE TO UPLOAD THE QUBO OR TO UPDATE PARAMETERS *****************

import sys
import dimod
from quanfluence_sdk import QuanfluenceClient

# ******* Signing in *************

client = QuanfluenceClient()
client.signin('rochester_user0','Rochesteruser@123')   # Preconfigured - DO NOT CHANGE

device_id = 16                              # Preconfigured - DO NOT CHANGE

# ******* Uploading the QUBO to the server *******

upload = client.upload_device_qubo(device_id, 'max_cut_triangle.qubo')   # ENTER THE PATH TO YOUR FILE HERE
filename = upload["result"]
print(filename)                             # REMEMBER THE FILENAME FOR EXECUTING IT ON THE DEVICE

# Function to get specifications of the Ising device being used

device = client.get_device(device_id)
print(device)

# Function to update parameteres of the Ising device as required
result = client.execute_device_qubo_file(device_id, filename)     # ENTER THE FILENAME OBTAINED AFTER UPLOADING THE QUBO
print(result)
