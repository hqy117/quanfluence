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

from quanfluence_sdk import QuanfluenceClient

# ******* Signing in *************

client = QuanfluenceClient()
client.signin('rochester_user0','Rochesteruser@123')   # Preconfigured - DO NOT CHANGE

device_id = 16                              # Preconfigured - DO NOT CHANGE

# ********** INSERT CODE FOR CREATING YOUR QUBO FROM HERE **************

# ***** An example QUBO is inlcuded in the following code remove if required

# ******** USER SECTION - ONLY PART THAT NEEDS TO BE CHANGED

Q = {(0, 0): 1, (0,1): -3, (1,1): 1}


# ********* END OF USER SECTION ************

# ****** End of user code **********

result = client.execute_device_qubo_input(device_id, Q)
print(result)
