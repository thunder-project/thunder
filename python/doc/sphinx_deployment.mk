# Copyright (c) Teracy, Inc. and individual contributors.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.

#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.

#     3. Neither the name of Teracy, Inc. nor the names of its contributors may be used
#        to endorse or promote products derived from this software without
#        specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Deployment configurations from sphinx_deployment project

# default deployment when $ make deploy
# deploy_gh_pages                            : to $ make deploy_gh_pages
# deploy_rsync                               : to $ make deploy_rsync
# deploy_heroku                              : to $ make deploy_heroku
# deploy_gh_pages deploy_rsync deploy_heroku : to $ make deploy_gh_pages then $ make deploy_rsync
#                                              and then $ make deploy_heroku
# default value: deploy_gh_pages
ifndef DEPLOY_DEFAULT
DEPLOY_DEFAULT = deploy_gh_pages
endif

# The deployment directory to be deployed
ifndef DEPLOY_DIR
DEPLOY_DIR      = _deploy
endif

# The heroku deployment directory to be deployed
# we must create this separated dir to avoid any conflict with _deploy (rsync and gh_pages)
ifndef DEPLOY_DIR_HEROKU
DEPLOY_DIR_HEROKU = _deploy_heroku
endif

# Copy contents from $(BUILDDIR) to $(DEPLOY_DIR)/$(DEPLOY_HTML_DIR) directory
ifndef DEPLOY_HTML_DIR
DEPLOY_HTML_DIR = docs-dev
endif


## -- Rsync Deploy config -- ##
# Be sure your public key is listed in your server's ~/.ssh/authorized_keys file
ifndef SSH_USER
SSH_USER       = user@domain.com
endif

ifndef SSH_PORT
SSH_PORT       = 22
endif

ifndef DOCUMENT_ROOT
DOCUMENT_ROOT  = ~/website.com/
endif

#If you choose to delete on sync, rsync will create a 1:1 match
ifndef RSYNC_DELETE
RSYNC_DELETE   = false
endif

# Any extra arguments to pass to rsync
ifndef RSYNC_ARGS
RSYNC_ARGS     =
endif

## -- Github Pages Deploy config -- ##

# Configure the right deployment branch
ifndef DEPLOY_BRANCH_GITHUB
DEPLOY_BRANCH_GITHUB = gh-pages
endif

#if REPO_URL_GITHUB was NOT defined by travis-ci
ifndef REPO_URL_GITHUB
# Configure your right github project repo
REPO_URL_GITHUB = git@github.com:thunder-project/thunder.git
endif

## -- Heroku Deployment Config -- ##

ifndef REPO_URL_HEROKU
# Configure your right heroku repo
# REPO_URL_HEROKU = git@heroku.com:spxd.git
endif


## end deployment configuration, don't edit anything below this line ##
#######################################################################

ifeq ($(RSYNC_DELETE), true)
RSYNC_DELETE_OPT = --delete
endif

init_gh_pages:
	@rm -rf $(DEPLOY_DIR)
	@mkdir -p $(DEPLOY_DIR)
	@cd $(DEPLOY_DIR); git init;\
		echo 'sphinx docs comming soon...' > index.html;\
		touch .nojekyll;\
		git add .; git commit -m "sphinx docs init";\
		git branch -m $(DEPLOY_BRANCH_GITHUB);\
		git remote add origin $(REPO_URL_GITHUB);
	@cd $(DEPLOY_DIR);\
		if ! git ls-remote origin $(DEPLOY_BRANCH_GITHUB) | grep $(DEPLOY_BRANCH_GITHUB) ; then \
			echo "Preparing Github deployment branch: $(DEPLOY_BRANCH_GITHUB) for the first time only...";\
			git push -u origin $(DEPLOY_BRANCH_GITHUB);\
		fi

setup_gh_pages: init_gh_pages
	@echo "Setting up gh-pages deployment..."
	@cd $(DEPLOY_DIR);\
		git fetch origin;\
		git reset --hard origin/$(DEPLOY_BRANCH_GITHUB);\
		git branch --set-upstream $(DEPLOY_BRANCH_GITHUB) origin/$(DEPLOY_BRANCH_GITHUB)
	@echo "Now you can deploy to Github Pages with 'make generate' and then 'make deploy_gh_pages'"

init_heroku:
	@rm -rf $(DEPLOY_DIR_HEROKU)
	@mkdir -p $(DEPLOY_DIR_HEROKU)
	@cd $(DEPLOY_DIR_HEROKU); git init;\
		cp -r ../.deploy_heroku/* .;\
		echo 'sphinx docs comming soon...' > public/index.html;\
		git add .; git commit -m "sphinx docs init";\
		git remote add origin $(REPO_URL_HEROKU);
	@cd $(DEPLOY_DIR_HEROKU);\
		if ! git ls-remote origin master | grep master ; then\
			echo "Preparing Heroku deployment for the first time only...";\
			git push -u origin master;\
		fi

setup_heroku: init_heroku
	@echo "setting up heroku deployment..."
	@cd $(DEPLOY_DIR_HEROKU);\
		git fetch origin;\
		git reset --hard origin/master;\
		git branch --set-upstream master origin/master
	@echo "Now you can deploy to Heroku with 'make generate' and then 'make deploy_heroku'"

generate: html

prepare_rsync_deployment:
	@echo "Preparing rsync deployment..."
	@mkdir -p $(DEPLOY_DIR)/$(DEPLOY_HTML_DIR)
	@echo "Copying files from '$(BUILDDIR)/html/.' to '$(DEPLOY_DIR)/$(DEPLOY_HTML_DIR)'"
	@cp -r $(BUILDDIR)/html/. $(DEPLOY_DIR)/$(DEPLOY_HTML_DIR)

deploy_rsync: prepare_rsync_deployment
	@echo "Deploying on rsync now..."
	rsync -avze 'ssh -p $(SSH_PORT)' --exclude-from $(realpath ./rsync_exclude) $(RSYNC_ARGS) $(RSYNC_DELETE_OPT) ${DEPLOY_DIR}/ $(SSH_USER):$(DOCUMENT_ROOT)

prepare_gh_pages_deployment:
	@echo "Preparing gh_pages deployment..."
	@echo "Pulling any updates from Github Pages..."
	@cd $(DEPLOY_DIR); git pull;
	@mkdir -p $(DEPLOY_DIR)/$(DEPLOY_HTML_DIR)
	@echo "Copying files from '$(BUILDDIR)/html/.' to '$(DEPLOY_DIR)/$(DEPLOY_HTML_DIR)'"
	@cp -r $(BUILDDIR)/html/. $(DEPLOY_DIR)/$(DEPLOY_HTML_DIR)

deploy_gh_pages: prepare_gh_pages_deployment
	@echo "Deploying on github pages now..."
	@cd $(DEPLOY_DIR); git add -A; git commit -m "docs updated at `date -u`";\
		git push origin $(DEPLOY_BRANCH) --quiet
	@echo "Github Pages deploy was completed at `date -u`"

prepare_heroku_deployment:
	@echo "Preparing heroku deployment..."
	@echo "Pulling any updates from Heroku..."
	@cd $(DEPLOY_DIR_HEROKU); git pull;
	@mkdir -p $(DEPLOY_DIR_HEROKU)/public/$(DEPLOY_HTML_DIR)
	@echo "Copying files from .deploy_heroku to $(DEPLOY_DIR_HEROKU)"
	@cp -r .deploy_heroku/. $(DEPLOY_DIR_HEROKU)
	@echo "Copying files from '$(BUILDDIR)/html/.' to '$(DEPLOY_DIR_HEROKU)/public/$(DEPLOY_HTML_DIR)'"
	@cp -r $(BUILDDIR)/html/. $(DEPLOY_DIR_HEROKU)/public/$(DEPLOY_HTML_DIR)


deploy_heroku: prepare_heroku_deployment
	@echo "Deploying on heroku now..."
	@cd $(DEPLOY_DIR_HEROKU); git add -A; git commit -m "docs updated at `date -u`";\
		git push origin master --quiet
	@echo "Heroku deployment was completed at `date -u`"


deploy: $(DEPLOY_DEFAULT)

gen_deploy: generate deploy
