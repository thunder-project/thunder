.. _deployment:

Deployment
==========
Running Spark and Thunder on Amazon's EC2 is an easy way to quickly leverage the computational power of a large cluster. The following instructions assume you've already installed Thunder as described in :ref:`install_local_ref`.

Setting up an Amazon account
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(You might be able to skip this step if you are already using EC2.) Go to `AWS <http://aws.amazon.com/>`_ to sign up for an account. Once you have created an account, go to `Identity and Access Management <https://console.aws.amazon.com/iam/#users>`_, select "Users" on the left, and click "Create new users" at the top. Follow the instructions to create a user for yourself. After you create a user, a window will pop up letting you "Show User Security Credentials". Click this to see your access key ID and secret access key (long strings of characters/numbers). Write these down, or click download to save them to a file, and close the window. While you have these handy, add the following two lines to your ``bash_profile``.

.. code-block:: bash

	export AWS_ACCESS_KEY_ID=<your-access-key-id>
	export AWS_SECRET_ACCESS_KEY=<your-secret-access-key>

Open a new terminal window so these changes take effect. Back on the AWS site, click the checkbox next to your user name. In the window that appears at the bottom of the page, select "Permissions > Attach user policy", select Administrator Access policy, and click Apply Policy. If you opened a brand new account it may take a couple hours for Amazon to verify it, so wait before proceeding.


Create your key pair
~~~~~~~~~~~~~~~~~~~~~~~
Go to the `Key Pair section on the EC2 console <https://console.aws.amazon.com/ec2/#KeyPairs:>`_. Click on the region name in the upper right (it will probably say Oregon) and select US East (N Virginia). Then click Create Key Pair at the top. Give it any name you want; we'll assume it's ``mykey``. You'll download a file called ``mykey.pem``; we'll assume you put it in your home folder. Set the correct permissions on this file by typing

.. code-block:: bash

	chmod 600 ~/mykey.pem

Launch a cluster
~~~~~~~~~~~~~~~~
Type the following command into the terminal

.. code-block:: bash

	thunder-ec2 -k mykey -i ~/mykey.pem -s <number-of-nodes> launch <cluster-name>

where ``mykey`` is the key-pair that we just created, ``<number-of-nodes>`` is the number of nodes in your cluster (try 2 just for testing), and ``<cluster-name>`` is just an identifier and can be anything. This command will take a few minutes. You should see lots of activity in the terminal. If the cluster doesn't start, which sometimes happens because the instances are slow to boot up (it will repeatedly report the error: ``ssh: connect to host... Connection refused``), wait for it to end then resume:

.. code-block:: bash

	thunder-ec2 -k mykey -i ~/mykey.pem launch <cluster-name> --resume

After it finishes you should see the message ``Cluster successfully launched!`` followed by a link to a URL. Go to this URL to see the Spark Web UI for your new cluster.

Login and start
~~~~~~~~~~~~~~~
To login just type

.. code-block:: bash

	thunder-ec2 -k mykey -i ~/mykey.pem login <cluster-name>

Once logged in, start the Spark shell in iPython:

.. code-block:: bash

	thunder

Remember to ``exit`` the cluster and shut it down when you are done!

.. code-block:: bash

	thunder-ec2 -k mykey -i ~/mykey.pem destroy <cluster-name>

Be careful, when you "destroy" a cluster, you forever lose any files or other information that you may have stored there.  If you simply want to pause your cluster so that you can return to using it later with its filesystems and data intact, you can instead "stop" and then "start" it again using these commands:

.. code-block:: bash

	thunder-ec2 -k mykey -i ~/mykey.pem stop <cluster-name>
	thunder-ec2 -k mykey -i ~/mykey.pem start <cluster-name>

When you stop a cluster, any data stored on the root partition ('/') will be there when you start it back up again.  (Watch out: data on scratch disks like /mnt and /mnt2 will not be saved!)  Be aware that Amazon will charge you a tiny fee to store these so-called Elastic Block Store (EBS) volumes when you are not using them.


Use the iPython notebook
~~~~~~~~~~~~~~~~~~~~~~~~

The iPython notebook is an especially useful way to do analyses interactively and look at results.  In order to connect to the iPython notebook with your web browser, you will need to establish access to the iPython server running on your EC2 master node.  There are two methods for doing this, described below.  Both methods work equally well, but one or the other may be better suited to your particular workflow so we leave it up to you to choose!

**Method 1: Connect directly over SSL**

This method allows you to connect directly to iPython notebook on EC2 over an encrypted (SSL) connection.  To do this, you will need to do one manual port configuration on the AWS console website. Go to the `EC2 dashboard <https://console.aws.amazon.com/ec2/v2/home>`_, click on "Security groups" in the list on the left, and find the name of your cluster in the list, and click on the entry "<cluster-name>-master". So if you called your cluster "test", look for "test-master". After selecting it, in the panel below, click the "Inbound" tab, click "Edit", click "Add rule", then type 8888 in the port range, and select "Anywhere" under source, then click "Save".

The rest is easy. Just log in to your cluster

.. code-block:: bash

	thunder-ec2 -k mykey -i ~/mykey.pem login <cluster-name>

If this is the first time you are logging in, you must run a script that configures iPython notebook to run in SSL mode.  Type:

.. code-block:: bash

	setup-notebook-ec2
	source /root/.bash_profile

During the script you will be asked to enter a password. Rememember what you give, as we'll need it again soon. At the end of the configuration you'll see the message ``iPython notebook successfully set up!`` followed by a link to a URL. If you now type:

.. code-block:: bash

	thunder

and go to the URL from the previous step in a web browser.  It will ask for the password we gave (if you get a message about SSL security, just click proceed). You are now running an iPython notebook server! Click ``New Notebook`` to start a session.  This URL will be accessible for as long as your cluster and iPython notebook are running.


**Method 2: Connect through an SSH Tunnel:**

This method uses the SSH protocol to establish a "tunnel" that routes network traffic from a port of your choosing on your local machine to and from the iPython notebook on the remote machine.  In essence, it makes it looks as though the remote iPython notebook is running on 'localhost' (i.e. the hostname of your local machine).

To connect to iPython notebok over an SSH tunnel, login to your cluster with

.. code-block:: bash

	thunder-ec2 -k mykey -i ~/mykey.pem login <cluster-name> --ssh-port-forwarding <local_port>:8888

A typical choice for <local_port> is 8888.  However, you may need to choose another port if, for example, port 8888 is already in use by another process.

If this is the first time you are logging in to your cluster, you must run a script that configures iPython notebook to run in SSH tunnel mode.  Type:

.. code-block:: bash

	setup-notebook-ec2-sshtunnel

Once you have done this, you are ready to run thunder!

.. code-block:: bash

	thunder

Simply point your browser at http://localhost:<local_port> and you will connect (over the SSH tunnel) to your iPython notebook server. Click ``New Notebook`` to start a session.

Note that although the SSH tunnel method involves less setup than the SSL method, it does require that you remain logged into your cluster whenever you wish to access the iPython notebook.  Once you log out, the SSH tunnel is disconnected. Of course, iPython is still running on your server, and you can access your running notebooks as soon as you log back in.  Just be sure to include the --ssh-port-forwarding option every time!

As a final word of caution: for now these two methods are mutually exclusive.  You can only use one of the above methods at a time.  However, if you change your mind simply remove the '/root/.ipython' directory on your EC2 master node and then follow the instruction above to switch methods.





