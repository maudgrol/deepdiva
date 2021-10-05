<template>
  <div class="container">
    <div class="header">
      <h1 class="header">Deep</h1>
      <img src="https://u-he.com/products/diva/assets/images/uhe-diva-logoheader.png">
    </div>

    <div class="center">
      <td colspan="7">
        <div class="file-upload-container p-5">
          <div
            v-if="noFiles"
            class="example-btn"
          >
            <h4>Drop a file anywhere to upload or</h4>
            <file-upload
              class="btn btn-primary"
              post-action="/upload/post"
              :drop="true"
              :drop-directory="true"
              v-model="files"
              ref="upload">
              <i class="fa fa-plus"></i>
              Select file
            </file-upload>
          </div>
          <div v-else>
            <h3>File: {{ files[0].name }}</h3>
            <b-form-group
              id="input-group-1"
              label="Start:"
              label-for="input-1"
              style="text-align: left;"
            >
              <b-form-input
                v-model="start"
                placeholder="where to start clip from"
                >
              </b-form-input>
            </b-form-group>
            <b-form-group
              id="input-group-1"
              label="End:"
              label-for="input-1"
              style="text-align: left;"
            >
              <b-form-input
                v-model="end"
                placeholder="where the clip ends"
                disabled
                >
              </b-form-input>
            </b-form-group>
            <b-form-group
              id="input-group-1"
              label="Output File Name:"
              label-for="input-1"
              style="text-align: left;"
            >
              <b-form-input
                v-model="filename"
                placeholder="the name of the output file"
                >
              </b-form-input>
            </b-form-group>
            <div class="buttons">
              <div style="margin-right: 5px;">
                <b-button
                  v-if="isLoading"
                  variant="primary"
                  disabled
                >
                  <b-spinner small type="grow"></b-spinner>
                  Loading...
                </b-button>
                <b-button
                  v-else
                  type="button"
                  variant="primary"
                  @click="getH2pFile"
                >
                  <i class="fa fa-arrow-up" aria-hidden="true"></i>
                  Submit
                </b-button>
              </div>
              <div>
                <b-button
                  type="button"
                  class="btn btn-danger"
                  :disable="isLoading"
                  @click.prevent="removeFile"
                >
                  <!-- @click.prevent="$refs.upload.active = false" -->
                  <i class="fa fa-stop" aria-hidden="true"></i>
                  Remove file
                </b-button>
              </div>
            </div>
          </div>
          <img
            v-if="fileDownloaded"
            :src="randomImage()"
            height="500px"
            width="500px"
            style="margin-top: 10px;"
          >
        </div>
      </td>
    </div>
  </div>
</template>

<script>
import FileUpload from 'vue-upload-component';
import axios from 'axios';

const PREDICTION_PATH = 'http://localhost:5000/prediction';

export default {
  components: {
    FileUpload,
  },
  data() {
    return {
      files: [],
      h2pFileFromServer: null,
      loading: false,
      start: '',
      fileDownloaded: false,
      filename: '',
      catImages: [
        'https://64.media.tumblr.com/a5d8b75916df802f6f1552b88d0cc84b/tumblr_nyjvtz4JBv1tvvm7oo1_640.gifv',
        'https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/c8e265e5-f5b3-48eb-a41b-55f40834e878/d9cyvam-08e79bf3-cd67-424c-8fe6-1b27842536ad.gif?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcL2M4ZTI2NWU1LWY1YjMtNDhlYi1hNDFiLTU1ZjQwODM0ZTg3OFwvZDljeXZhbS0wOGU3OWJmMy1jZDY3LTQyNGMtOGZlNi0xYjI3ODQyNTM2YWQuZ2lmIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.1tNPz7Faa458EFIUpya4ddXIlcWJ0KiBYXTApDTc-_4',
        'https://i.pinimg.com/originals/57/fd/91/57fd913e381100821b94b4314fafa73f.gif',
        'https://c.tenor.com/wm7HkUKYS4YAAAAC/cat-bongo.gif',
        'https://c.tenor.com/JFVJNv-nmEcAAAAC/space-synth.gif',
        'https://media3.giphy.com/media/ArAgo5dU2z2xO/giphy.gif',
        'https://laughingsquid.com/wp-content/uploads/2020/08/Bongo-Cat-in-Space.gif',
        'https://media0.giphy.com/media/iO1SXgxGMJnTxCmAKs/giphy-downsized-large.gif',
        'https://c.tenor.com/OWm4qgjMe2AAAAAM/space-cat.gif',
        'https://64.media.tumblr.com/5a9e02c6e62044009299ffc73cc1fccf/tumblr_mmlp9cs04j1r4xjo2o1_500.gifv',
      ],
    };
  },
  computed: {
    noFiles() {
      return this.files.length < 1;
    },
    isLoading() {
      return this.loading;
    },
    end() {
      if (this.start) {
        return parseInt(this.start, 0) + 2;
      }

      return '';
    },
  },
  methods: {
    randomImage() {
      const randomNumber = Math.floor(Math.random() * this.catImages.length);
      return this.catImages[randomNumber];
    },
    removeFile() {
      this.files = [];
      this.fileDownloaded = false;
      this.start = '';
      this.filename = '';
    },
    downloadFile(response) {
      const fileURL = window.URL.createObjectURL(new Blob([response.data]));
      const fileLink = document.createElement('a');

      fileLink.href = fileURL;
      const filename = this.filename ? `${this.filename}.h2p` : 'h2p_file.h2p';
      fileLink.setAttribute('download', filename);
      document.body.appendChild(fileLink);

      fileLink.click();
    },
    createDataAndConfig() {
      // create wav file to send to backend
      const data = new FormData();
      const file = new Blob([this.files[0].file]);

      data.append('wavfile', file);
      data.append('start', this.start);

      const config = {
        headers: { 'Content-Type': 'multipart/form-data' },
      };

      return [data, config];
    },
    async getH2pFile() {
      this.loading = true;

      try {
        if (this.noFiles || this.start === '') {
          // eslint-disable-next-line
          alert("Please upload a file and choose a start time");
        } else {
          const [data, config] = this.createDataAndConfig();

          await axios.post(PREDICTION_PATH, data, config)
            .then((response) => {
              this.downloadFile(response);
              this.fileDownloaded = true;
            })
            .catch((error) => {
              // eslint-disable-next-line
              alert(error.response.data["Message"])
            });
        }
      } catch (error) {
        // eslint-disable-next-line
        alert(error);
      } finally {
        this.loading = false;
      }
    },
  },
};
</script>

<style>
.file-upload-container {
  min-width: 616px;
}

.header {
  display: flex;
  justify-content: center;
}

.input {
  text-align: center;
}

.buttons {
  display: flex;
}

.center {
  display: flex;
  justify-content: center;
}

.name {
  display: flex;
  flex-direction: column;
  justify-items: center;
  justify-content: center;
}

.header {
  text-align: center;
  margin-top: 30px;
}

.container label.btn {
  margin-bottom: 0;
  margin-right: 1rem;
}

.container .drop-active {
  top: 0;
  bottom: 0;
  right: 0;
  left: 0;
  position: fixed;
  z-index: 9999;
  opacity: .6;
  text-align: center;
  background: #000;
}

.container .drop-active h3 {
  margin: -.5em 0 0;
  position: absolute;
  top: 50%;
  left: 0;
  right: 0;
  -webkit-transform: translateY(-50%);
  -ms-transform: translateY(-50%);
  transform: translateY(-50%);
  font-size: 40px;
  color: #fff;
  padding: 0;
}
</style>
