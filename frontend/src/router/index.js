import Vue from 'vue';
import Router from 'vue-router';
import Books from '../components/Books.vue';
import DeepDiva from '../components/DeepDiva.vue';
import Ping from '../components/Ping.vue';

Vue.use(Router);

export default new Router({
  mode: 'history',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '/',
      name: 'DeepDiva',
      component: DeepDiva,
    },
    {
      path: '/Books',
      name: 'Books',
      component: Books,
    },
    {
      path: '/ping',
      name: 'Ping',
      component: Ping,
    },
  ],
});
