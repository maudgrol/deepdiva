<template>
  <div class="container">
    <h1 class="header">Deep Diva</h1>

    <div class="center">
      <td colspan="7">
        <div class="text-center p-5">
          <h4>Drop a file anywhere to upload<br/>or</h4>
          <div
            v-if="noFiles"
            class="example-btn"
          >
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
            <h3 class="name-header">File being uploaded: {{ files[0].name }}</h3>
            <div class="buttons">
              <div>
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
                  Start Upload
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
            <b-form-input
              v-model="start"
              placeholder="where to start clip from"
              >
            </b-form-input>
          </div>
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
    };
  },
  computed: {
    noFiles() {
      return this.files.length < 1;
    },
    isLoading() {
      return this.loading;
    },
  },
  methods: {
    removeFile() {
      this.files = [];
    },
    async getH2pFile() {
      this.loading = true;

      try {
        if (this.noFiles || this.start === '') {
          // eslint-disable-next-line
          alert("Please upload a file and choose a start time");
        } else {
          // eslint-disable-next-line
          console.log(this.files)

          // create wav file to send to backend
          const data = new FormData();
          const file = new Blob(this.files);

          data.append('wavfile', file, file.name);
          data.append('start', this.start);

          const config = {
            headers: { 'content-type': 'multipart/form-data' },
          };

          const payload = {
            wavfile: file,
            start: 100,
          };

          // send wav file to the backend and process h2p file download
          await axios.post(PREDICTION_PATH, payload, config)
            .then((response) => {
              const fileURL = window.URL.createObjectURL(new Blob([response.data]));
              const fileLink = document.createElement('a');

              fileLink.href = fileURL;
              fileLink.setAttribute('download', 'h2p_file.h2p');
              document.body.appendChild(fileLink);

              fileLink.click();
            })
            .catch((error) => {
              // eslint-disable-next-line
              console.error(error);
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
.input {
  text-align: center;
}

.buttons {
  display: flex;
  justify-content: center;
}

.center {
  display: flex;
  justify-content: center;
}

.name-header {
  text-align: center;
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
